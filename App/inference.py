"""
[3] INFERENCE LAYER — GABUNGAN (versi teman kamu + fitur upload/retrain)
============================================================================

Cara pakai dari app.py:
    import inference
    inference.set_data_source("default")   # pakai model bawaan
    inference.set_data_source("custom")    # pakai model hasil upload user
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# PATH ARTIFACT — DEFAULT (dataset bawaan) vs CUSTOM (hasil upload user)
# ------------------------------------------------------------------
DEFAULT_PATHS = {
    "als": os.path.join(BASE_DIR, "..", "Model", "ALS", "als_model.pkl"),
    "bundling": os.path.join(BASE_DIR, "..", "artifacts", "association_table.pkl"),
    "forecast": os.path.join(BASE_DIR, "..", "artifacts", "demand_forecast.pkl"),
    "dim_game": os.path.join(BASE_DIR, "..", "Forecast", "game_example.csv"),
}

CUSTOM_ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts_custom")
CUSTOM_PATHS = {
    "als": os.path.join(CUSTOM_ARTIFACT_DIR, "als_model.pkl"),
    "bundling": os.path.join(CUSTOM_ARTIFACT_DIR, "association_table.pkl"),
    "forecast": os.path.join(CUSTOM_ARTIFACT_DIR, "demand_forecast.pkl"),
    "dim_game": os.path.join(CUSTOM_ARTIFACT_DIR, "dim_game.csv"),
}

if "data_source" not in st.session_state:
    st.session_state["data_source"] = "default"


def set_data_source(source: str):
    """source: 'default' atau 'custom'. Otomatis clear cache supaya artifact di-load ulang."""
    st.session_state["data_source"] = source
    load_bundling_artifacts.clear()
    load_als_artifacts.clear()
    load_forecast_artifacts.clear()
    load_genre_lookup.clear()


def get_data_source() -> str:
    return st.session_state.get("data_source", "default")


def _paths() -> dict:
    return CUSTOM_PATHS if get_data_source() == "custom" else DEFAULT_PATHS


def custom_artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in CUSTOM_PATHS.values())


def _detect_genre_columns(dim_game_df: pd.DataFrame) -> list:
    """Deteksi otomatis kolom genre (one-hot 0/1), TIDAK di-hardcode, supaya kompatibel
    dengan skema genre dataset manapun (termasuk dataset custom upload user)."""
    exclude = {"game_name", "publisher", "platform", "game_id", "price"}
    genre_cols = []
    for col in dim_game_df.columns:
        if col in exclude:
            continue
        unique_vals = set(dim_game_df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            genre_cols.append(col)
    return genre_cols


# ============================================================================
# 1. BUNDLING / CROSS-SELL
# ============================================================================
@st.cache_resource
def load_bundling_artifacts(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_bundling_suggestions(game_name: str, top_n: int = 5) -> dict:
    """Input: nama game (harus persis/exact match dari dataset).
    Output: dict berisi persentase kelakuan + daftar rekomendasi bundling."""
    data = load_bundling_artifacts(_paths()["bundling"])
    association = data["association"]
    total_users = data["total_users"]
    popularity = data["game_popularity"]

    if game_name not in popularity:
        return {"error": f"Game '{game_name}' tidak ditemukan di data."}

    jumlah_pembeli = popularity[game_name]
    persen_laku = (jumlah_pembeli / total_users) * 100
    rekomendasi = association.get(game_name, [])[:top_n]

    return {
        "game": game_name,
        "persen_laku": round(persen_laku, 2),
        "jumlah_pembeli": jumlah_pembeli,
        "rekomendasi_bundling": rekomendasi,
    }


def get_all_game_names() -> list:
    """Mengambil semua daftar nama game yang diurutkan untuk auto-suggest di UI."""
    data = load_bundling_artifacts(_paths()["bundling"])
    return sorted(list(data["game_popularity"].keys()))


def search_game_names(query: str, limit: int = 15) -> list:
    """Cari nama game yang mengandung 'query' dengan sorting relevansi & popularitas."""
    data = load_bundling_artifacts(_paths()["bundling"])
    popularity = data["game_popularity"]
    all_games = list(popularity.keys())
    query_lower = query.strip().lower()
    if not query_lower:
        return []
    matches = [g for g in all_games if query_lower in g.lower()]
    matches.sort(key=lambda g: (not g.lower().startswith(query_lower), -popularity.get(g, 0), len(g)))
    return matches[:limit]


# ============================================================================
# 2. NEXT-ITEM RECOMMENDATION (ALS) + XAI 3-LAPIS
# ============================================================================
@st.cache_resource
def load_als_artifacts(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_all_user_ids() -> list:
    """Mengambil semua daftar User ID yang valid dan terurut untuk auto-suggest di UI."""
    artifacts = load_als_artifacts(_paths()["als"])
    user_to_idx = artifacts.get("user_to_idx", {})
    return [int(u) for u in sorted(user_to_idx.keys())]


def search_user_ids(query, limit: int = 15) -> list:
    """Cari User ID yang berawalan atau mengandung query angka."""
    all_users = get_all_user_ids()
    query_str = str(query).strip()
    if not query_str:
        return all_users[:limit]
    matches = [u for u in all_users if str(u).startswith(query_str)]
    if len(matches) < limit:
        additional = [u for u in all_users if query_str in str(u) and u not in matches]
        matches.extend(additional)
    return matches[:limit]


def _cosine_sim(vec_a, vec_b_matrix):
    """Cosine similarity antara 1 vektor vs banyak vektor sekaligus (untuk cari item paling mirip)."""
    norm_a = vec_a / (np.linalg.norm(vec_a) + 1e-9)
    norm_b = vec_b_matrix / (np.linalg.norm(vec_b_matrix, axis=1, keepdims=True) + 1e-9)
    return norm_b @ norm_a


@st.cache_resource
def load_genre_lookup(path: str):
    """Load tabel genre per game (dari dim_game) -> dipakai untuk cek genre overlap di XAI."""
    dim_game = pd.read_csv(path)
    genre_cols = _detect_genre_columns(dim_game)
    dim_game = dim_game.drop_duplicates(subset="game_name").set_index("game_name")
    return dim_game[genre_cols]


def _get_genres(game_name: str) -> set:
    """Ambil daftar genre (nama kolom yang bernilai 1) untuk 1 game."""
    genre_table = load_genre_lookup(_paths()["dim_game"])
    if game_name not in genre_table.index:
        return set()
    row = genre_table.loc[game_name]
    return set(row[row == 1].index)


def _genre_overlap(game_a: str, game_b: str) -> dict:
    """Cek genre yang sama persis antara 2 game -> bukti pendukung independen dari skor ALS."""
    genres_a = _get_genres(game_a)
    genres_b = _get_genres(game_b)
    common = genres_a & genres_b
    return {"genre_sama": sorted(common), "ada_genre_sama": len(common) > 0}


def _co_purchase_rate(reference_idx: int, recommended_idx: int, user_item_matrix) -> dict:
    """
    Hitung: dari semua user yang punya game REFERENSI, berapa persen yang JUGA
    punya game yang direkomendasikan. Ini murni statistik dari data user lain,
    independen dari perhitungan similarity ALS -> bukti pendukung kedua.
    """
    owners_reference = set(user_item_matrix[:, reference_idx].nonzero()[0])
    owners_recommended = set(user_item_matrix[:, recommended_idx].nonzero()[0])

    if len(owners_reference) == 0:
        return {"co_purchase_pct": 0.0, "jumlah_owner_referensi": 0}

    overlap = owners_reference & owners_recommended
    pct = (len(overlap) / len(owners_reference)) * 100
    return {
        "co_purchase_pct": round(pct, 1),
        "jumlah_owner_referensi": len(owners_reference),
        "jumlah_beli_keduanya": len(overlap),
    }


def explain_recommendation(user_id, recommended_game_idx: int) -> dict:
    """
    XAI untuk ALS, 3 lapis:
      1. Cari game DI HISTORI USER yang vektornya paling mirip (cosine similarity)
         dengan game yang direkomendasikan -> "game referensi".
      2. Cek APAKAH kemiripan itu didukung oleh genre yang sama (genre overlap).
      3. Cek APAKAH kemiripan itu didukung oleh pola pembelian user lain (co-purchase rate).

    CATATAN: genre overlap & co-purchase adalah BUKTI PENDUKUNG yang berkorelasi
    dengan skor similarity, bukan penyebab langsung skor itu.
    """
    artifacts = load_als_artifacts(_paths()["als"])
    model = artifacts["model"]
    user_item_matrix = artifacts["user_item_matrix"]
    user_to_idx = artifacts["user_to_idx"]
    idx_to_game = artifacts["idx_to_game"]

    uidx = user_to_idx[user_id]
    owned_indices = user_item_matrix[uidx].nonzero()[1]
    if len(owned_indices) == 0:
        return {"alasan": "Tidak ada histori untuk dibandingkan."}

    item_factors = model.item_factors
    rec_vector = item_factors[recommended_game_idx]
    owned_vectors = item_factors[owned_indices]

    sims = _cosine_sim(rec_vector, owned_vectors)
    best_idx_in_owned = np.argmax(sims)
    reference_idx = owned_indices[best_idx_in_owned]
    most_similar_game = idx_to_game[owned_indices[best_idx_in_owned]]
    recommended_game = idx_to_game[recommended_game_idx]
    similarity_pct = round(float(sims[best_idx_in_owned]) * 100, 1)

    genre_info = _genre_overlap(most_similar_game, recommended_game)
    co_purchase_info = _co_purchase_rate(reference_idx, recommended_game_idx, user_item_matrix)

    faktor_pendukung = []
    if genre_info["ada_genre_sama"]:
        genre_str = ", ".join(genre_info["genre_sama"])
        faktor_pendukung.append(f"sama-sama genre {genre_str}")
    if co_purchase_info["co_purchase_pct"] > 0:
        faktor_pendukung.append(
            f"{co_purchase_info['co_purchase_pct']}% dari user lain yang main "
            f"'{most_similar_game}' juga main game ini"
        )
    if faktor_pendukung:
        penjelasan_tambahan = " Didukung oleh: " + "; ".join(faktor_pendukung) + "."
    else:
        penjelasan_tambahan = (
            " Tidak ada kesamaan genre/pola pembelian bersama yang jelas -- "
            "kemiripan ini murni dari pola preferensi tersembunyi yang dipelajari model."
        )

    return {
        "alasan": f"Mirip dengan '{most_similar_game}' yang pernah kamu mainkan (kemiripan preferensi {similarity_pct}%).{penjelasan_tambahan}",
        "game_referensi": most_similar_game,
        "tingkat_kemiripan": similarity_pct,
        "genre_sama": genre_info["genre_sama"],
        "co_purchase_pct": co_purchase_info["co_purchase_pct"],
    }


def get_next_item_recommendations(user_id, k: int = 10, with_explanation: bool = True) -> dict:
    """Input: user_id.
    Output: histori transaksi + rekomendasi game berikutnya beserta skor DAN alasan (XAI)."""
    artifacts = load_als_artifacts(_paths()["als"])
    model = artifacts["model"]
    user_item_matrix = artifacts["user_item_matrix"]
    user_to_idx = artifacts["user_to_idx"]
    idx_to_game = artifacts["idx_to_game"]

    if user_id not in user_to_idx:
        return {"error": f"User ID '{user_id}' tidak ditemukan di data."}

    uidx = user_to_idx[user_id]
    owned_indices = user_item_matrix[uidx].nonzero()[1]
    histori = [idx_to_game[i] for i in owned_indices]

    game_indices, scores = model.recommend(uidx, user_item_matrix[uidx], N=k, filter_already_liked_items=True)
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores_pct = (scores - scores.min()) / (scores.max() - scores.min()) * 100
    else:
        scores_pct = np.zeros_like(scores)

    rekomendasi = []
    for i, p in zip(game_indices, scores_pct):
        item = {"game": idx_to_game[i], "skor_kemiripan": round(float(p), 1)}
        if with_explanation:
            item["xai"] = explain_recommendation(user_id, i)
        rekomendasi.append(item)

    return {
        "user_id": user_id,
        "jumlah_histori": len(histori),
        "histori_transaksi": histori,
        "rekomendasi": rekomendasi,
    }


# ============================================================================
# 3. DEMAND FORECASTING
# ============================================================================
@st.cache_resource
def load_forecast_artifacts(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_trending_genres(top_n: int = 5) -> dict:
    data = load_forecast_artifacts(_paths()["forecast"])
    trending = data["trending_summary"]
    return {
        "trending_naik": trending.head(top_n).to_dict("records"),
        "trending_turun": trending.tail(top_n).to_dict("records"),
    }


def get_genre_forecast(genre: str):
    data = load_forecast_artifacts(_paths()["forecast"])
    return data["forecast_by_genre"].get(genre)


def get_all_genres() -> list:
    data = load_forecast_artifacts(_paths()["forecast"])
    return sorted(data["forecast_by_genre"].keys())


def explain_forecast(genre: str) -> dict:
    forecast_df = get_genre_forecast(genre)
    if forecast_df is None:
        return {"error": f"Genre '{genre}' tidak ditemukan."}

    future_only = forecast_df[forecast_df["is_forecast"]]
    trend_awal = forecast_df[~forecast_df["is_forecast"]]["trend"].iloc[0]
    trend_akhir = future_only["trend"].iloc[-1]
    trend_pct_change = ((trend_akhir - trend_awal) / trend_awal) * 100 if trend_awal != 0 else 0
    avg_yearly_effect = future_only["yearly"].mean()

    if abs(trend_pct_change) > abs(avg_yearly_effect / forecast_df["yhat"].mean() * 100):
        dominan = "trend jangka panjang"
    else:
        dominan = "pola musiman (yearly seasonality)"

    return {
        "genre": genre,
        "perubahan_trend_persen": round(trend_pct_change, 2),
        "rata2_efek_musiman": round(float(avg_yearly_effect), 2),
        "faktor_dominan": dominan,
        "penjelasan": (
            f"Forecast '{genre}' didorong terutama oleh {dominan}. "
            f"Trend jangka panjang berubah {trend_pct_change:+.1f}% dari awal data ke periode forecast, "
            f"sementara efek musiman rata-rata menyumbang {avg_yearly_effect:+.1f} transaksi/bulan."
        ),
    }
