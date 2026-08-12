"""
ALS (Alternating Least Squares) Recommender — Prediksi "barang apa yang akan dibeli user berikutnya"
========================================================================================================
Model collaborative filtering untuk implicit feedback. Berbeda dari popularity model,
ALS mempelajari POLA PERSONAL tiap user lewat matrix factorization: user & item
direpresentasikan sebagai vektor laten, dan skor rekomendasi = dot product keduanya.

Requirement: pip install implicit scipy pandas numpy

Cara pakai:
    python als_recommender.py
"""

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

pd.set_option("display.width", 120)
np.random.seed(42)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSAKSI_PATH = os.path.join(SCRIPT_DIR, r"C:\Users\User\Downloads\Compfest\Model\transaction.csv")
DIM_GAME_PATH = os.path.join(SCRIPT_DIR, r"C:\Users\User\Downloads\Compfest\Model\game.csv")

trans = pd.read_csv(TRANSAKSI_PATH)
dim_game = pd.read_csv(DIM_GAME_PATH)

trans["date_time"] = pd.to_datetime(trans["date_time"])
trans = trans.sort_values(["user_id", "date_time"]).reset_index(drop=True)

print(f"Total transaksi : {len(trans):,}")
print(f"Total user      : {trans['user_id'].nunique():,}")
print(f"Total game      : {trans['game_name'].nunique():,}")

# ------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (leave-last-out, SAMA seperti popularity model
#    supaya hasil evaluasi bisa dibandingkan apple-to-apple)
# ------------------------------------------------------------------
trans["rank_desc"] = trans.groupby("user_id")["date_time"].rank(method="first", ascending=False)
test = trans[trans["rank_desc"] == 1].copy()
train = trans[trans["rank_desc"] > 1].copy()

print(f"\nTrain: {len(train):,} transaksi | Test: {len(test):,} transaksi (1 per user)")

# ------------------------------------------------------------------
# 3. ENCODE user_id & game_name JADI INDEX INTEGER (0..n-1)
#    ALS butuh matrix sparse, jadi tiap user/game perlu id numerik berurutan
# ------------------------------------------------------------------
user_ids = train["user_id"].unique()
game_names = train["game_name"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
idx_to_user = {i: u for u, i in user_to_idx.items()}

game_to_idx = {g: i for i, g in enumerate(game_names)}
idx_to_game = {i: g for g, i in game_to_idx.items()}

train["user_idx"] = train["user_id"].map(user_to_idx)
train["game_idx"] = train["game_name"].map(game_to_idx)

# ------------------------------------------------------------------
# 4. HITUNG CONFIDENCE SCORE (bobot implicit feedback)
#    Semakin tinggi rating & playtime -> semakin yakin user memang suka game ini.
#    rating dinormalisasi 0-1, playtime di-log supaya outlier tidak mendominasi.
# ------------------------------------------------------------------
train["rating_norm"] = (train["rating"] - 1) / 4  # rating 1-5 -> 0-1
train["playtime_log"] = np.log1p(train["playtime_hours"])
train["playtime_norm"] = train["playtime_log"] / train["playtime_log"].max()

# confidence = 1 (base, karena sudah beli/main) + bobot rating + bobot playtime
ALPHA = 4.0  # seberapa besar pengaruh rating & playtime terhadap confidence
train["confidence"] = 1 + ALPHA * (0.6 * train["rating_norm"] + 0.4 * train["playtime_norm"])

# ------------------------------------------------------------------
# 5. BUILD SPARSE USER-ITEM MATRIX
# ------------------------------------------------------------------
n_users = len(user_to_idx)
n_games = len(game_to_idx)

user_item_matrix = sp.csr_matrix(
    (train["confidence"], (train["user_idx"], train["game_idx"])),
    shape=(n_users, n_games),
)

print(f"\nMatrix shape: {user_item_matrix.shape} | density: {user_item_matrix.nnz / (n_users * n_games):.6f}")

# ------------------------------------------------------------------
# 6. TRAIN MODEL ALS
# ------------------------------------------------------------------
model = AlternatingLeastSquares(
    factors=64,          # dimensi vektor laten user/item
    regularization=0.05, # cegah overfitting
    iterations=20,
    random_state=42,
)

print("\nTraining ALS...")
model.fit(user_item_matrix)
print("Training selesai.")

# ------------------------------------------------------------------
# 7. FUNGSI REKOMENDASI
# ------------------------------------------------------------------
def recommend_for_user(user_id: int, k: int = 10) -> list:
    """Rekomendasikan top-k game untuk user_id berdasarkan model ALS."""
    if user_id not in user_to_idx:
        return []  # cold-start user, tidak ada di training data

    uidx = user_to_idx[user_id]
    game_indices, scores = model.recommend(
        uidx, user_item_matrix[uidx], N=k, filter_already_liked_items=True
    )
    return [idx_to_game[i] for i in game_indices]


# ------------------------------------------------------------------
# 8. EVALUASI: Hit Rate@K & NDCG@K (metodologi sama dengan popularity model)
# ------------------------------------------------------------------
def evaluate(k: int = 10):
    hits, ndcgs = [], []
    for _, row in test.iterrows():
        uid = row["user_id"]
        actual_game = row["game_name"]

        recs = recommend_for_user(uid, k=k)
        if actual_game in recs:
            hits.append(1)
            rank = recs.index(actual_game) + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            hits.append(0)
            ndcgs.append(0)

    return np.mean(hits), np.mean(ndcgs)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EVALUASI MODEL ALS (Hit Rate@K & NDCG@K)")
    print("=" * 60)

    for k in [5, 10, 20]:
        hr, ndcg = evaluate(k=k)
        print(f"K={k:2d}  ->  HitRate@{k}: {hr:.4f} | NDCG@{k}: {ndcg:.4f}")

    # Contoh rekomendasi untuk 1 user
    contoh_user = int(train["user_id"].iloc[0])
    print(f"\nContoh rekomendasi ALS untuk user_id={contoh_user}:")
    for g in recommend_for_user(contoh_user, k=10):
        print(f"  - {g}")

    # Simpan model & mapping supaya bisa dipakai lagi tanpa training ulang
    import pickle
    artifacts = {
        "model": model,
        "user_item_matrix": user_item_matrix,
        "user_to_idx": user_to_idx,
        "idx_to_game": idx_to_game,
    }
    out_path = os.path.join(SCRIPT_DIR, "als_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nModel disimpan ke: {out_path}")
