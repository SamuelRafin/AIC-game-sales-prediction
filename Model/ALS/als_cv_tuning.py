"""
Fine-Tuning ALS dengan Temporal Cross-Validation
============================================================================
Sebelumnya kita cuma pakai 1x train/test split (leave-last-out) -> hasilnya
bisa saja sedikit noisy/beruntung-kebetulan. Di sini kita pakai ROLLING
TEMPORAL CROSS-VALIDATION: evaluasi di 3 titik waktu berbeda per user
(transaksi terakhir, kedua-terakhir, ketiga-terakhir), lalu rata-ratakan.

Ini menghormati sifat data yang time-ordered (tidak boleh random split
biasa untuk task next-item prediction), sambil tetap dapat estimasi
performa yang lebih robust dibanding 1 split saja.

Requirement: pip install implicit scipy pandas numpy

Cara pakai:
    python als_cv_tuning.py
"""

import os
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

pd.set_option("display.width", 120)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSAKSI_PATH = os.path.join(SCRIPT_DIR, r"C:\Users\User\Downloads\Compfest\Model\Dataset_TerminalGame_Mentah - transaksi.csv")

trans = pd.read_csv(TRANSAKSI_PATH)
trans["date_time"] = pd.to_datetime(trans["date_time"])
trans = trans.sort_values(["user_id", "date_time"]).reset_index(drop=True)
trans["rank_desc"] = trans.groupby("user_id")["date_time"].rank(method="first", ascending=False)

trans["rating_norm"] = (trans["rating"] - 1) / 4
trans["playtime_log"] = np.log1p(trans["playtime_hours"])
trans["playtime_norm"] = trans["playtime_log"] / trans["playtime_log"].max()

N_FOLDS = 3  # fold 1 = leave-last-out, fold 2 = leave-2nd-last-out, fold 3 = leave-3rd-last-out


def make_fold(fold_i: int):
    """fold_i: transaksi dengan rank_desc == fold_i jadi test, sisanya (rank_desc > fold_i) jadi train."""
    test = trans[trans["rank_desc"] == fold_i].copy()
    train = trans[trans["rank_desc"] > fold_i].copy()

    user_ids = train["user_id"].unique()
    game_names = train["game_name"].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    game_to_idx = {g: i for i, g in enumerate(game_names)}

    train = train.copy()
    train["user_idx"] = train["user_id"].map(user_to_idx)
    train["game_idx"] = train["game_name"].map(game_to_idx)

    test_user_idx = test["user_id"].map(user_to_idx)
    test_game_idx = test["game_name"].map(game_to_idx)
    valid = test_user_idx.notna() & test_game_idx.notna()
    test_user_idx = test_user_idx[valid].astype(int).values
    test_game_idx = test_game_idx[valid].astype(int).values

    n_users, n_games = len(user_to_idx), len(game_to_idx)
    return train, n_users, n_games, test_user_idx, test_game_idx


def build_confidence_matrix(train, n_users, n_games, alpha_conf: float) -> sp.csr_matrix:
    conf = 1 + alpha_conf * (0.6 * train["rating_norm"] + 0.4 * train["playtime_norm"])
    return sp.csr_matrix((conf, (train["user_idx"], train["game_idx"])), shape=(n_users, n_games))


def evaluate_matrix(scores, owned_mask, test_user_idx, test_game_idx, k=10):
    scores = np.where(owned_mask, -np.inf, scores)
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-row_scores, axis=1)
    topk_idx = np.take_along_axis(part, order, axis=1)

    hits, ndcgs = [], []
    for u_idx, g_idx in zip(test_user_idx, test_game_idx):
        rec_row = topk_idx[u_idx]
        pos = np.where(rec_row == g_idx)[0]
        if len(pos) > 0:
            rank = pos[0] + 1
            hits.append(1)
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            hits.append(0)
            ndcgs.append(0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


# ------------------------------------------------------------------
# GRID (dipilih untuk cover ruang parameter yang belum dites: iterations
# lebih tinggi, kombinasi factors/regularization/alpha_conf lainnya)
# ------------------------------------------------------------------
grid = [
    {"factors": 64, "regularization": 0.05, "alpha_conf": 4, "iterations": 20},   # baseline
    {"factors": 128, "regularization": 0.05, "alpha_conf": 4, "iterations": 20},
    {"factors": 64, "regularization": 0.1, "alpha_conf": 8, "iterations": 20},
    {"factors": 128, "regularization": 0.01, "alpha_conf": 8, "iterations": 30},
    {"factors": 64, "regularization": 0.05, "alpha_conf": 4, "iterations": 50},   # iterations lebih tinggi
    {"factors": 96, "regularization": 0.03, "alpha_conf": 6, "iterations": 30},
]

if __name__ == "__main__":
    print("=" * 80)
    print(f"FINE-TUNING ALS dengan {N_FOLDS}-FOLD TEMPORAL CROSS-VALIDATION")
    print("=" * 80)

    # Precompute fold data sekali di awal (dipakai berulang untuk semua config)
    fold_data = {}
    for fold_i in range(1, N_FOLDS + 1):
        fold_data[fold_i] = make_fold(fold_i)
        print(f"Fold {fold_i} siap: train={len(fold_data[fold_i][0]):,} baris")

    results = []
    for cfg in grid:
        t0 = time.time()
        fold_hr, fold_ndcg = [], []

        for fold_i in range(1, N_FOLDS + 1):
            train, n_users, n_games, test_user_idx, test_game_idx = fold_data[fold_i]
            ui_matrix = build_confidence_matrix(train, n_users, n_games, cfg["alpha_conf"])

            model = AlternatingLeastSquares(
                factors=cfg["factors"],
                regularization=cfg["regularization"],
                iterations=cfg["iterations"],
                random_state=42,
            )
            model.fit(ui_matrix)

            scores = model.user_factors @ model.item_factors.T
            owned_mask = ui_matrix.toarray() > 0
            hr, ndcg = evaluate_matrix(scores, owned_mask, test_user_idx, test_game_idx, k=10)
            fold_hr.append(hr)
            fold_ndcg.append(ndcg)

        elapsed = time.time() - t0
        label = f"f={cfg['factors']} reg={cfg['regularization']} alpha={cfg['alpha_conf']} iters={cfg['iterations']}"
        mean_hr, std_hr = np.mean(fold_hr), np.std(fold_hr)
        mean_ndcg = np.mean(fold_ndcg)
        results.append({
            "config": label,
            "mean_hit_rate@10": mean_hr,
            "std_hit_rate@10": std_hr,
            "mean_ndcg@10": mean_ndcg,
            "fold_hit_rates": [round(x, 4) for x in fold_hr],
            "time_s": round(elapsed, 1),
        })
        print(f"\n{label}")
        print(f"  Fold HR@10: {[round(x,4) for x in fold_hr]}")
        print(f"  Mean HR@10: {mean_hr:.4f} (+/- {std_hr:.4f}) | Mean NDCG@10: {mean_ndcg:.4f} | {elapsed:.1f}s")

    results_df = pd.DataFrame(results).sort_values("mean_hit_rate@10", ascending=False)
    print("\n" + "=" * 80)
    print("RINGKASAN (diurutkan dari Mean Hit Rate@10 terbaik)")
    print("=" * 80)
    print(results_df.drop(columns=["fold_hit_rates"]).to_string(index=False))

    out_path = os.path.join(SCRIPT_DIR, "als_cv_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nHasil disimpan ke: {out_path}")
