"""
[2] MODEL TRAINING - Bundling / Cross-Sell Association
============================================================================
Berbeda dari versi sebelumnya (game_association.py) yang menghitung on-the-fly
tiap kali dipanggil, script ini adalah TRUE OFFLINE TRAINING STEP:
  - Hitung SEMUA pasangan game yang punya genre sama & pernah dibeli bersama
  - Simpan hasilnya sebagai artifact (association_table.csv + .pkl)
  - Dashboard/inference layer tinggal LOAD hasil ini, tidak perlu hitung ulang

Logika bundling: kalau banyak pembeli game A yang JUGA membeli game B
(dengan genre yang sama), maka A & B kandidat kuat untuk dibundling.

Output metrik utama per pasangan (A -> B):
  - persen_laku_A         : % dari total user yang membeli game A
  - jumlah_beli_keduanya  : jumlah user yang beli A DAN B
  - persen_bundling       : P(beli B | sudah beli A) dalam persen
                             = jumlah_beli_keduanya / jumlah_pembeli_A * 100

Requirement: pip install pandas numpy scipy

Cara pakai:
    python train_bundling.py
Output:
    ../artifacts/association_table.csv
    ../artifacts/association_table.pkl   (dict, lebih cepat di-load dashboard)
"""

import os
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle

pd.set_option("display.width", 120)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..")  # naik 1 folder dari ALS/ ke Model/, tempat CSV berada
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "..", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRANSAKSI_PATH = os.path.join(DATA_DIR, r"C:\Users\User\Downloads\Compfest\Model\transaction.csv")
DIM_GAME_PATH = os.path.join(DATA_DIR, r"C:\Users\User\Downloads\Compfest\Model\game.csv")

GENRE_COLS = [
    "Action", "Adult", "Adventure", "Arcade", "Beat 'Em Up", "Brain Training",
    "Card & Board Game", "Casual", "Educational", "Family", "Fighting", "Fitness",
    "Hack And Slash", "Horror", "Indie", "Moba", "Music", "Party", "Pinball",
    "Platform", "Point-And-Click", "Puzzle", "Quiz", "Racing",
    "Real Time Strategy (Rts)", "Rhythm", "Shooter", "Simulation", "Simulator",
    "Sport", "Sports", "Strategy", "Tactical", "Trivia",
    "Turn-Based Strategy (Tbs)", "Unique", "Unknown", "Visual Novel", "RPG",
]

MIN_CO_BUYERS = 3     # minimal user yang beli keduanya, biar tidak nyangkut ke kebetulan (misal cuma 1 user)
TOP_N_PER_GAME = 15   # simpan berapa banyak rekomendasi bundling per game


def main():
    t_start = time.time()
    print("=" * 70)
    print("TRAINING: Bundling / Cross-Sell Association Model")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    trans = pd.read_csv(TRANSAKSI_PATH)
    dim_game = pd.read_csv(DIM_GAME_PATH)
    print(f"Transaksi : {len(trans):,} baris")
    print(f"Game      : {trans['game_name'].nunique():,} unik")

    total_users = trans["user_id"].nunique()

    # ------------------------------------------------------------------
    # 2. ENCODE game_name & user_id -> index integer (untuk sparse matrix)
    # ------------------------------------------------------------------
    game_names = trans["game_name"].unique()
    game_to_idx = {g: i for i, g in enumerate(game_names)}
    idx_to_game = {i: g for g, i in game_to_idx.items()}
    n_games = len(game_to_idx)

    user_ids = trans["user_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    n_users = len(user_to_idx)

    trans["game_idx"] = trans["game_name"].map(game_to_idx)
    trans["user_idx"] = trans["user_id"].map(user_to_idx)

    # ------------------------------------------------------------------
    # 3. BANGUN USER-ITEM MATRIX BINER (1 kalau user pernah beli game itu)
    # ------------------------------------------------------------------
    ui_matrix = sp.csr_matrix(
        (np.ones(len(trans)), (trans["user_idx"], trans["game_idx"])),
        shape=(n_users, n_games),
    )
    ui_matrix.data[:] = 1  # kalau ada duplikat, tetap dianggap 1 (bukan double count)

    item_popularity = np.asarray(ui_matrix.sum(axis=0)).flatten()  # jumlah pembeli per game

    # ------------------------------------------------------------------
    # 4. CO-OCCURRENCE MATRIX (item-item): berapa user yang beli A DAN B
    #    Dihitung via perkalian matrix (jauh lebih cepat dari loop manual)
    # ------------------------------------------------------------------
    print("\nMenghitung co-occurrence matrix (item x item)...")
    co_occurrence = (ui_matrix.T @ ui_matrix).tocoo()  # sparse (n_games x n_games)
    print(f"Jumlah pasangan game yang pernah dibeli bersama: {co_occurrence.nnz:,}")

    # ------------------------------------------------------------------
    # 5. GENRE MATRIX -> untuk filter "genre sama"
    # ------------------------------------------------------------------
    dim_game_indexed = dim_game.drop_duplicates(subset="game_name").set_index("game_name")
    game_order = [idx_to_game[i] for i in range(n_games)]
    genre_matrix = dim_game_indexed.reindex(game_order)[GENRE_COLS].fillna(0).to_numpy(dtype=bool)

    # ------------------------------------------------------------------
    # 6. SUSUN TABEL HASIL: filter pasangan (A,B) dengan genre overlap
    #    dan jumlah co-buyer >= MIN_CO_BUYERS
    # ------------------------------------------------------------------
    print("\nMenyusun tabel asosiasi (filter genre overlap + min co-buyers)...")
    rows = []
    coo_rows, coo_cols, coo_vals = co_occurrence.row, co_occurrence.col, co_occurrence.data

    for a, b, co_count in zip(coo_rows, coo_cols, coo_vals):
        if a == b:
            continue
        if co_count < MIN_CO_BUYERS:
            continue
        overlap = genre_matrix[a] & genre_matrix[b]
        if not overlap.any():
            continue

        game_a, game_b = idx_to_game[a], idx_to_game[b]
        overlap_genres = [g for g, ok in zip(GENRE_COLS, overlap) if ok]
        persen_bundling = (co_count / item_popularity[a]) * 100

        rows.append({
            "game_A": game_a,
            "game_B": game_b,
            "genre_sama": ", ".join(overlap_genres),
            "persen_laku_A": round((item_popularity[a] / total_users) * 100, 2),
            "jumlah_pembeli_A": int(item_popularity[a]),
            "jumlah_pembeli_B": int(item_popularity[b]),
            "jumlah_beli_keduanya": int(co_count),
            "persen_bundling": round(persen_bundling, 2),
        })

    association_df = pd.DataFrame(rows)
    print(f"Total baris asosiasi (sebelum top-N filter): {len(association_df):,}")

    # Ambil top-N rekomendasi bundling terbaik per game_A saja (biar file tidak raksasa)
    association_df = (
        association_df.sort_values(["game_A", "persen_bundling"], ascending=[True, False])
        .groupby("game_A")
        .head(TOP_N_PER_GAME)
        .reset_index(drop=True)
    )
    print(f"Total baris asosiasi (setelah top-{TOP_N_PER_GAME} per game): {len(association_df):,}")
    print(f"Cakupan: {association_df['game_A'].nunique():,} game punya rekomendasi bundling")

    # ------------------------------------------------------------------
    # 7. SIMPAN ARTIFACT
    # ------------------------------------------------------------------
    csv_path = os.path.join(ARTIFACT_DIR, "association_table.csv")
    association_df.to_csv(csv_path, index=False)
    print(f"\nDisimpan: {csv_path}")

    # Versi pkl: dict {game_A: [daftar rekomendasi]} -> lookup O(1) super cepat untuk dashboard
    association_dict = {}
    for game_a, group in association_df.groupby("game_A"):
        association_dict[game_a] = group.drop(columns=["game_A"]).to_dict("records")

    pkl_path = os.path.join(ARTIFACT_DIR, "association_table.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "association": association_dict,
            "total_users": total_users,
            "game_popularity": {idx_to_game[i]: int(item_popularity[i]) for i in range(n_games)},
        }, f)
    print(f"Disimpan: {pkl_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal waktu training: {elapsed:.1f} detik")

    # ------------------------------------------------------------------
    # 8. CONTOH HASIL (sanity check)
    # ------------------------------------------------------------------
    contoh_game = trans["game_name"].value_counts().index[0]
    print(f"\nContoh hasil untuk game paling populer: '{contoh_game}'")
    if contoh_game in association_dict:
        for item in association_dict[contoh_game][:5]:
            print(f"  {item['persen_bundling']:5.2f}% -> {item['game_B']} (genre: {item['genre_sama']})")
    else:
        print("  (tidak ada kandidat bundling yang lolos filter untuk game ini)")


if __name__ == "__main__":
    main()
