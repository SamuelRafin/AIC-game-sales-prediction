"""
Full Data Completion Pipeline (IGDB-first, statistik sebagai fallback)
========================================================================
Urutan prioritas pengisian missing value untuk tiap kolom:

  genre, publisher, platform, release_date, metacritic_score
  1) Cari dulu ke IGDB API (data ASLI dari database game)
  2) Kalau IGDB tidak punya -> fallback statistik:
       - metacritic_score : regresi dari playstation_score, lalu median
                             per (platform, genre)
       - genre / publisher: "Unknown"
       - platform / release_date : baris di-drop HANYA kalau IGDB juga
                             tidak ketemu sama sekali (harusnya sangat sedikit)

Cara pakai:
1. Isi CLIENT_ID / CLIENT_SECRET di bagian CONFIG (atau environment variable)
2. pip install requests pandas scikit-learn
3. python full_data_completion_pipeline.py
   (proses akan cache tiap hasil pencarian IGDB, aman dijalankan ulang
   kalau terhenti di tengah jalan)
"""

import requests
import pandas as pd
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")

# =========================== CONFIG ===========================

CLIENT_ID = os.environ.get("IGDB_CLIENT_ID", "vib2w2r7hmmas56wri7ltp5bima6e0")
CLIENT_SECRET = os.environ.get("IGDB_CLIENT_SECRET", "jxrm3qzj5ie9qw94himvopgsd3zung")

INPUT_CSV = "game_details.csv"
OUTPUT_CSV = "game_details_final_no_missing.csv"

TOKEN_CACHE_FILE = "igdb_token_cache.json"
SEARCH_CACHE_FILE = "igdb_full_search_cache.json"

REQUEST_DELAY = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 5
SAVE_EVERY_N = 20

PLATFORM_ALIASES = {
    "ps3": "PlayStation 3", "ps4": "PlayStation 4", "ps5": "PlayStation 5",
    "ps vita": "PlayStation Vita", "psp": "PlayStation Portable",
}

# ================================================================


def get_igdb_token():
    if Path(TOKEN_CACHE_FILE).exists():
        with open(TOKEN_CACHE_FILE, "r") as f:
            cached = json.load(f)
        if cached.get("expires_at", 0) > time.time():
            return cached["access_token"]

    url = "https://id.twitch.tv/oauth2/token"
    params = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "client_credentials"}
    r = requests.post(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    token_data = {"access_token": data["access_token"], "expires_at": time.time() + data.get("expires_in", 3600) - 60}
    with open(TOKEN_CACHE_FILE, "w") as f:
        json.dump(token_data, f)
    return token_data["access_token"]


def load_cache(path):
    return json.load(open(path)) if Path(path).exists() else {}


def save_cache(path, cache):
    json.dump(cache, open(path, "w"), indent=2)


def query_igdb_full(game_name, client_id, token):
    """Ambil genre, publisher, platform, tanggal rilis, dan rating dari IGDB."""
    url = "https://api.igdb.com/v4/games"
    headers = {"Client-ID": client_id, "Authorization": f"Bearer {token}"}
    safe_name = game_name.replace('"', "")
    body = (
        f'search "{safe_name}"; '
        f'fields name, genres.name, platforms.name, first_release_date, '
        f'aggregated_rating, involved_companies.company.name, '
        f'involved_companies.publisher; '
        f'limit 5;'
    )
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, data=body, timeout=15)
            if r.status_code == 401:
                token = get_igdb_token()
                headers["Authorization"] = f"Bearer {token}"
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}")
            time.sleep(RETRY_DELAY)
    return []


def extract_publisher(item):
    for ic in item.get("involved_companies", []):
        if ic.get("publisher") and ic.get("company", {}).get("name"):
            return ic["company"]["name"]
    return None


def extract_platform(item, target_platform=None):
    plats = [p.get("name") for p in item.get("platforms", []) if p.get("name")]
    if not plats:
        return None
    if target_platform:
        for p in plats:
            if target_platform.lower() in p.lower():
                return p
    return plats[0]


def extract_release_date(item):
    ts = item.get("first_release_date")
    if not ts:
        return None
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}"


def pick_best_match(results, target_platform=None):
    if not results:
        return None

    def has_platform(item):
        if not target_platform:
            return True
        plats = [p.get("name", "").lower() for p in item.get("platforms", [])]
        return any(target_platform.lower() in p for p in plats)

    candidates = [r for r in results if has_platform(r) and r.get("aggregated_rating")]
    if candidates:
        return candidates[0]
    candidates = [r for r in results if has_platform(r)]
    if candidates:
        return candidates[0]
    return results[0]


def main():
    df = pd.read_csv(INPUT_CSV, sep=";")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # bersihkan placeholder & tipe data
    for col in ["genre", "publisher"]:
        df[col] = df[col].replace("--", pd.NA)
    for col in ["metacritic_score", "playstation_score", "highest_price"]:
        cleaned = df[col].astype("object").astype(str).str.replace("€", "", regex=False).str.strip()
        cleaned = cleaned.replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    df["genre_source"] = "original"
    df["publisher_source"] = "original"
    df["platform_source"] = "original"
    df["release_date_source"] = "original"
    df["metacritic_source"] = "original"

    token = get_igdb_token()
    cache = load_cache(SEARCH_CACHE_FILE)

    needs_lookup = df[
        df["genre"].isna() | df["publisher"].isna() | df["platform"].isna()
        | df["release_date"].isna() | df["metacritic_score"].isna()
    ].index.tolist()

    print(f"Total baris yang perlu dicek ke IGDB: {len(needs_lookup)}")

    for i, idx in enumerate(needs_lookup, start=1):
        game_name = str(df.at[idx, "game_name"]).strip()
        platform_raw = str(df.at[idx, "platform"]).strip().lower() if pd.notna(df.at[idx, "platform"]) else None
        target_platform = PLATFORM_ALIASES.get(platform_raw, platform_raw)

        cache_key = game_name
        if cache_key in cache:
            best = cache[cache_key]
        else:
            print(f"[{i}/{len(needs_lookup)}] Cari: {game_name}")
            results = query_igdb_full(game_name, CLIENT_ID, token)
            best = pick_best_match(results, target_platform)
            cache[cache_key] = best
            time.sleep(REQUEST_DELAY)

        if not best:
            continue

        if pd.isna(df.at[idx, "genre"]):
            genres = [g["name"] for g in best.get("genres", []) if g.get("name")]
            if genres:
                df.at[idx, "genre"] = " / ".join(genres)
                df.at[idx, "genre_source"] = "igdb"

        if pd.isna(df.at[idx, "publisher"]):
            pub = extract_publisher(best)
            if pub:
                df.at[idx, "publisher"] = pub
                df.at[idx, "publisher_source"] = "igdb"

        if pd.isna(df.at[idx, "platform"]):
            plat = extract_platform(best)
            if plat:
                df.at[idx, "platform"] = plat
                df.at[idx, "platform_source"] = "igdb"

        if pd.isna(df.at[idx, "release_date"]):
            rd = extract_release_date(best)
            if rd:
                df.at[idx, "release_date"] = rd
                df.at[idx, "release_date_source"] = "igdb"

        if pd.isna(df.at[idx, "metacritic_score"]) and best.get("aggregated_rating"):
            df.at[idx, "metacritic_score"] = round(best["aggregated_rating"])
            df.at[idx, "metacritic_source"] = "igdb"

        if i % SAVE_EVERY_N == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            save_cache(SEARCH_CACHE_FILE, cache)
            print(f"  -> progres disimpan ({i}/{len(needs_lookup)})")

    save_cache(SEARCH_CACHE_FILE, cache)

    # ---------- FALLBACK STATISTIK (hanya untuk yang IGDB juga tidak punya) ----------

    train = df[df["metacritic_score"].notna() & df["playstation_score"].notna()]
    if len(train) > 10:
        reg = LinearRegression().fit(train[["playstation_score"]], train["metacritic_score"])
        mask = df["metacritic_score"].isna() & df["playstation_score"].notna()
        est = (reg.coef_[0] * df.loc[mask, "playstation_score"] + reg.intercept_).clip(0, 100).round()
        df.loc[mask, "metacritic_score"] = est
        df.loc[mask, "metacritic_source"] = "estimated_from_playstation_score"

    mask_still_missing_meta = df["metacritic_score"].isna()
    df["metacritic_score"] = df.groupby(["platform", "genre"])["metacritic_score"].transform(lambda s: s.fillna(s.median()))
    df["metacritic_score"] = df.groupby("platform")["metacritic_score"].transform(lambda s: s.fillna(s.median()))
    df["metacritic_score"] = df["metacritic_score"].fillna(df["metacritic_score"].median())
    df.loc[mask_still_missing_meta, "metacritic_source"] = "imputed_median"

    df["genre"] = df["genre"].fillna("Unknown")
    df["publisher"] = df["publisher"].fillna("Unknown")

    df["highest_price"] = df.groupby(["platform", "genre"])["highest_price"].transform(lambda s: s.fillna(s.median()))
    df["highest_price"] = df["highest_price"].fillna(df["highest_price"].median())

    # baris yang platform/release_date TETAP kosong walau sudah dicoba IGDB -> baru di-drop
    before_final = len(df)
    df = df.dropna(subset=["platform", "release_date"])
    dropped = before_final - len(df)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== RINGKASAN AKHIR ===")
    print(f"Total baris final       : {len(df)}")
    print(f"Baris di-drop (terakhir): {dropped}  (platform/release_date tetap tidak ketemu di manapun)")
    print("\nSumber data per kolom:")
    for col in ["genre_source", "publisher_source", "platform_source", "release_date_source", "metacritic_source"]:
        print(f"\n{col}:")
        print(df[col].value_counts())
    print(f"\nMissing value tersisa:\n{df.isna().sum()}")
    print(f"\nHasil disimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
