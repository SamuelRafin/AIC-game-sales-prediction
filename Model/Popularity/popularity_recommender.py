"""
Popularity-Based Recommender System — Prediksi "barang apa yang akan dibeli user berikutnya"
================================================================================================
Model ini adalah BASELINE: merekomendasikan game yang paling populer secara umum
(atau populer per genre favorit user), tanpa personalisasi individual.
Tujuannya jadi pembanding sebelum lanjut ke model yang lebih canggih (ALS, LightFM, dst).

Cara pakai:
    python popularity_recommender.py
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 120)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
TRANSAKSI_PATH = r"C:\Users\User\Downloads\Compfest\Model\Dataset_TerminalGame_Mentah - transaksi.csv"
DIM_GAME_PATH = r"C:\Users\User\Downloads\Compfest\Model\Dataset_TerminalGame_Mentah - dim_game.csv"

trans = pd.read_csv(TRANSAKSI_PATH)
dim_game = pd.read_csv(DIM_GAME_PATH)

trans["date_time"] = pd.to_datetime(trans["date_time"])
trans = trans.sort_values(["user_id", "date_time"]).reset_index(drop=True)

print(f"Total transaksi : {len(trans):,}")
print(f"Total user      : {trans['user_id'].nunique():,}")
print(f"Total game      : {trans['game_name'].nunique():,}")

# ------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (leave-last-out, per user, berdasarkan waktu)
# ------------------------------------------------------------------
# Transaksi TERAKHIR tiap user -> test set (yang mau kita prediksi)
# Transaksi lainnya           -> train set (histori yang model "lihat")
trans["rank_desc"] = trans.groupby("user_id")["date_time"].rank(
    method="first", ascending=False
)

test = trans[trans["rank_desc"] == 1].copy()
train = trans[trans["rank_desc"] > 1].copy()

print(f"\nTrain: {len(train):,} transaksi | Test: {len(test):,} transaksi (1 per user)")

# ------------------------------------------------------------------
# 3. BUILD POPULARITY SCORE (dari data TRAIN saja)
# ------------------------------------------------------------------
# Skor popularitas = kombinasi jumlah pembelian + rata-rata rating.
# Rating dipakai sebagai pembobot supaya game populer TAPI rating jelek
# tidak dianggap "bagus" untuk direkomendasikan.
popularity = (
    train.groupby("game_name")
    .agg(
        jumlah_pembelian=("user_id", "count"),
        rata_rating=("rating", "mean"),
        rata_playtime=("playtime_hours", "mean"),
    )
    .reset_index()
)

# Bayesian-average rating supaya game dengan sedikit transaksi tidak
# tiba-tiba unggul cuma karena kebetulan rating-nya 5 dari 1 pembelian.
C = popularity["jumlah_pembelian"].mean()          # rata2 jumlah pembelian semua game
M = popularity["rata_rating"].mean()                # rata2 rating semua game
popularity["skor_populer"] = (
    (popularity["jumlah_pembelian"] / (popularity["jumlah_pembelian"] + C)) * popularity["rata_rating"]
    + (C / (popularity["jumlah_pembelian"] + C)) * M
) * np.log1p(popularity["jumlah_pembelian"])  # log agar jumlah pembelian tetap berpengaruh

popularity = popularity.sort_values("skor_populer", ascending=False).reset_index(drop=True)

print("\nTop 10 game paling populer (dari train set):")
print(popularity[["game_name", "jumlah_pembelian", "rata_rating", "skor_populer"]].head(10).to_string(index=False))

# ------------------------------------------------------------------
# 4. GENRE-AWARE POPULARITY (opsional, lebih dari sekadar global top-N)
# ------------------------------------------------------------------
# Manfaatkan user_profile_genres di data transaksi supaya rekomendasi
# sedikit lebih relevan: ranking populer HANYA di antara game yang
# genre-nya overlap dengan genre favorit user.
game_genre_lookup = dim_game.set_index("game_name")["genre"].to_dict()

def genre_set(genre_str: str) -> set:
    if pd.isna(genre_str):
        return set()
    return set(g.strip().lower() for g in str(genre_str).replace("|", "/").split("/"))

popularity["genre_set"] = popularity["game_name"].map(game_genre_lookup).apply(genre_set)


def recommend_for_user(user_id: int, k: int = 10, use_genre_filter: bool = True) -> list:
    """Rekomendasikan top-k game untuk user_id, exclude game yang sudah pernah dibeli di TRAIN."""
    already_owned = set(train.loc[train["user_id"] == user_id, "game_name"])

    candidates = popularity[~popularity["game_name"].isin(already_owned)].copy()

    if use_genre_filter:
        user_row = train.loc[train["user_id"] == user_id, "user_profile_genres"]
        if len(user_row) > 0:
            user_genres = genre_set(user_row.iloc[0])
            if user_genres:
                mask = candidates["genre_set"].apply(lambda gs: len(gs & user_genres) > 0)
                filtered = candidates[mask]
                # kalau hasil filter genre terlalu sedikit, fallback ke popularity global
                if len(filtered) >= k:
                    candidates = filtered

    return candidates["game_name"].head(k).tolist()


# ------------------------------------------------------------------
# 5. EVALUASI: apakah game yang BENAR dibeli next ada di rekomendasi?
# ------------------------------------------------------------------
def evaluate(k: int = 10, use_genre_filter: bool = True, sample_users: int = None):
    test_users = test["user_id"].unique()
    if sample_users:
        test_users = np.random.choice(test_users, size=sample_users, replace=False)

    hits, ndcgs = [], []
    for uid in test_users:
        actual_game = test.loc[test["user_id"] == uid, "game_name"].iloc[0]
        recs = recommend_for_user(uid, k=k, use_genre_filter=use_genre_filter)

        if actual_game in recs:
            hits.append(1)
            rank = recs.index(actual_game) + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            hits.append(0)
            ndcgs.append(0)

    hit_rate = np.mean(hits)
    ndcg = np.mean(ndcgs)
    return hit_rate, ndcg


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EVALUASI MODEL (Hit Rate@K & NDCG@K)")
    print("=" * 60)

    for k in [5, 10, 20]:
        hr_global, ndcg_global = evaluate(k=k, use_genre_filter=False)
        hr_genre, ndcg_genre = evaluate(k=k, use_genre_filter=True)
        print(f"\nK={k}")
        print(f"  Popularity Global     -> HitRate@{k}: {hr_global:.4f} | NDCG@{k}: {ndcg_global:.4f}")
        print(f"  Popularity + Genre    -> HitRate@{k}: {hr_genre:.4f} | NDCG@{k}: {ndcg_genre:.4f}")

    # Contoh rekomendasi untuk 1 user
    contoh_user = int(train["user_id"].iloc[0])
    print(f"\nContoh rekomendasi untuk user_id={contoh_user}:")
    for g in recommend_for_user(contoh_user, k=10):
        print(f"  - {g}")

    # Simpan tabel popularity full untuk referensi
    out_path = "popularity_ranking.csv"
    popularity.drop(columns=["genre_set"]).to_csv(out_path, index=False)
    print(f"\nTabel ranking popularitas disimpan ke: {out_path}")
