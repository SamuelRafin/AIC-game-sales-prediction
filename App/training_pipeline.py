"""
TRAINING PIPELINE — versi callable (bisa dipanggil langsung dari dashboard)
============================================================================
Refactor dari train_bundling.py, train_demand_forecast.py, als_recommender.py
supaya bisa dipanggil sebagai FUNGSI (bukan cuma dijalankan sebagai script
terpisah), dengan DataFrame yang sudah di-load (misal dari file yang di-upload
user di Streamlit) sebagai input, bukan cuma dari path file tetap.

3 fungsi utama:
  - train_bundling_model(transaksi_df, dim_game_df, artifact_dir)
  - train_forecast_model(transaksi_df, dim_game_df, artifact_dir)
  - train_als_model(transaksi_df, artifact_dir)

Semua fungsi ini MENULIS artifact ke folder yang ditentukan (artifact_dir),
dengan nama file yang KONSISTEN dengan yang dibaca inference.py, supaya
dashboard bisa langsung load hasilnya begitu training selesai.

Requirement: pip install pandas numpy scipy implicit prophet scikit-learn
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")


def detect_genre_columns(dim_game_df: pd.DataFrame) -> list:
    """
    Deteksi otomatis kolom genre (one-hot 0/1) di dim_game, TIDAK di-hardcode,
    supaya kompatibel dengan dataset customer manapun (kolom genre bisa beda-beda).
    Kriteria: kolom numerik yang isinya cuma 0 dan 1, dan bukan kolom identitas
    (game_name, publisher, platform, dll).
    """
    exclude = {"game_name", "publisher", "platform", "game_id"}
    genre_cols = []
    for col in dim_game_df.columns:
        if col in exclude:
            continue
        unique_vals = set(dim_game_df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            genre_cols.append(col)
    return genre_cols


def validate_transaksi_schema(transaksi_df: pd.DataFrame) -> list:
    """Cek kolom wajib ada di data transaksi. Return list error (kosong kalau valid)."""
    required = ["user_id", "game_name", "date_time"]
    missing = [c for c in required if c not in transaksi_df.columns]
    errors = []
    if missing:
        errors.append(f"Kolom wajib tidak ditemukan di data transaksi: {missing}")
    return errors


def validate_dim_game_schema(dim_game_df: pd.DataFrame) -> list:
    """Cek kolom wajib ada di data dim_game. Return list error (kosong kalau valid)."""
    errors = []
    if "game_name" not in dim_game_df.columns:
        errors.append("Kolom 'game_name' tidak ditemukan di data dim_game.")
    genre_cols = detect_genre_columns(dim_game_df)
    if len(genre_cols) == 0:
        errors.append("Tidak ada kolom genre (one-hot 0/1) yang terdeteksi di data dim_game.")
    return errors


# ============================================================================
# 1. TRAIN BUNDLING (Association)
# ============================================================================
def train_bundling_model(transaksi_df: pd.DataFrame, dim_game_df: pd.DataFrame,
                          artifact_dir: str, min_co_buyers: int = 3, top_n_per_game: int = 15,
                          progress_callback=None) -> dict:
    os.makedirs(artifact_dir, exist_ok=True)
    genre_cols = detect_genre_columns(dim_game_df)

    if progress_callback:
        progress_callback("Menyiapkan data...", 0.1)

    game_names = transaksi_df["game_name"].unique()
    game_to_idx = {g: i for i, g in enumerate(game_names)}
    idx_to_game = {i: g for g, i in game_to_idx.items()}
    n_games = len(game_to_idx)

    user_ids = transaksi_df["user_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    n_users = len(user_to_idx)

    total_users = transaksi_df["user_id"].nunique()

    df = transaksi_df.copy()
    df["game_idx"] = df["game_name"].map(game_to_idx)
    df["user_idx"] = df["user_id"].map(user_to_idx)

    ui_matrix = sp.csr_matrix(
        (np.ones(len(df)), (df["user_idx"], df["game_idx"])), shape=(n_users, n_games)
    )
    ui_matrix.data[:] = 1
    item_popularity = np.asarray(ui_matrix.sum(axis=0)).flatten()

    if progress_callback:
        progress_callback("Menghitung co-occurrence antar game...", 0.4)

    co_occurrence = (ui_matrix.T @ ui_matrix).tocoo()

    dim_game_indexed = dim_game_df.drop_duplicates(subset="game_name").set_index("game_name")
    game_order = [idx_to_game[i] for i in range(n_games)]
    genre_matrix = dim_game_indexed.reindex(game_order)[genre_cols].fillna(0).to_numpy(dtype=bool)

    if progress_callback:
        progress_callback("Menyusun tabel asosiasi...", 0.7)

    rows = []
    for a, b, co_count in zip(co_occurrence.row, co_occurrence.col, co_occurrence.data):
        if a == b or co_count < min_co_buyers:
            continue
        overlap = genre_matrix[a] & genre_matrix[b]
        if not overlap.any():
            continue
        overlap_genres = [g for g, ok in zip(genre_cols, overlap) if ok]
        rows.append({
            "game_A": idx_to_game[a], "game_B": idx_to_game[b],
            "genre_sama": ", ".join(overlap_genres),
            "persen_laku_A": round((item_popularity[a] / total_users) * 100, 2),
            "jumlah_pembeli_A": int(item_popularity[a]), "jumlah_pembeli_B": int(item_popularity[b]),
            "jumlah_beli_keduanya": int(co_count),
            "persen_bundling": round((co_count / item_popularity[a]) * 100, 2),
        })

    association_df = pd.DataFrame(rows)
    if len(association_df) > 0:
        association_df = (
            association_df.sort_values(["game_A", "persen_bundling"], ascending=[True, False])
            .groupby("game_A").head(top_n_per_game).reset_index(drop=True)
        )

    association_dict = {}
    for game_a, group in association_df.groupby("game_A"):
        association_dict[game_a] = group.drop(columns=["game_A"]).to_dict("records")

    artifact = {
        "association": association_dict,
        "total_users": total_users,
        "game_popularity": {idx_to_game[i]: int(item_popularity[i]) for i in range(n_games)},
    }
    with open(os.path.join(artifact_dir, "association_table.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    if progress_callback:
        progress_callback("Bundling selesai.", 1.0)
    return artifact


# ============================================================================
# 2. TRAIN DEMAND FORECAST (Prophet per genre)
# ============================================================================
def train_forecast_model(transaksi_df: pd.DataFrame, dim_game_df: pd.DataFrame,
                          artifact_dir: str, forecast_months: int = 6, min_monthly_avg: int = 5,
                          progress_callback=None) -> dict:
    from prophet import Prophet

    os.makedirs(artifact_dir, exist_ok=True)
    genre_cols = detect_genre_columns(dim_game_df)

    if progress_callback:
        progress_callback("Menyiapkan data time-series...", 0.1)

    df = transaksi_df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["month"] = df["date_time"].dt.to_period("M").dt.to_timestamp()

    dim_game_indexed = dim_game_df.drop_duplicates(subset="game_name").set_index("game_name")
    genre_lookup = dim_game_indexed[genre_cols]
    df = df.merge(genre_lookup, left_on="game_name", right_index=True, how="left")
    df[genre_cols] = df[genre_cols].fillna(0)

    genre_monthly = {}
    for genre in genre_cols:
        sub = df[df[genre] == 1]
        if len(sub) == 0:
            continue
        monthly_counts = sub.groupby("month").size().reset_index(name="y")
        monthly_counts.rename(columns={"month": "ds"}, inplace=True)
        genre_monthly[genre] = monthly_counts

    all_forecasts, trending_summary, skipped = [], [], []
    n_genres = len(genre_monthly)

    for idx, (genre, df_genre) in enumerate(genre_monthly.items()):
        if progress_callback:
            progress_callback(f"Training forecast: {genre}...", 0.2 + 0.7 * (idx / max(n_genres, 1)))

        avg_monthly = df_genre["y"].mean()
        if avg_monthly < min_monthly_avg or len(df_genre) < 12:
            skipped.append(genre)
            continue

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.80)
        model.fit(df_genre)
        future = model.make_future_dataframe(periods=forecast_months, freq="MS")
        forecast = model.predict(future)

        component_cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        if "yearly" in forecast.columns:
            component_cols.append("yearly")
        forecast_out = forecast[component_cols].copy()
        if "yearly" not in forecast_out.columns:
            forecast_out["yearly"] = 0.0
        forecast_out["genre"] = genre
        forecast_out["is_forecast"] = forecast_out["ds"] > df_genre["ds"].max()
        all_forecasts.append(forecast_out)

        actual_recent = df_genre.sort_values("ds")["y"].tail(3).mean()
        forecast_future = forecast_out[forecast_out["is_forecast"]]["yhat"].mean()
        pct_change = ((forecast_future - actual_recent) / actual_recent) * 100 if actual_recent > 0 else 0
        trending_summary.append({
            "genre": genre, "rata2_actual_3bulan_terakhir": round(actual_recent, 1),
            "rata2_forecast_kedepan": round(forecast_future, 1), "persen_perubahan": round(pct_change, 2),
            "total_historis": int(df_genre["y"].sum()),
        })

    if progress_callback:
        progress_callback("Menyimpan hasil forecast...", 0.95)

    forecast_df = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    trending_df = pd.DataFrame(trending_summary).sort_values("persen_perubahan", ascending=False) if trending_summary else pd.DataFrame()

    artifact = {
        "forecast_by_genre": {g: d for g, d in forecast_df.groupby("genre")} if len(forecast_df) else {},
        "trending_summary": trending_df,
        "forecast_months": forecast_months,
        "skipped_genres": skipped,
    }
    with open(os.path.join(artifact_dir, "demand_forecast.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    if progress_callback:
        progress_callback("Forecasting selesai.", 1.0)
    return artifact


# ============================================================================
# 3. TRAIN ALS (Next-Item Recommendation)
# ============================================================================
def train_als_model(transaksi_df: pd.DataFrame, artifact_dir: str,
                     factors: int = 64, regularization: float = 0.05, iterations: int = 20,
                     alpha_conf: float = 4.0, progress_callback=None) -> dict:
    from implicit.als import AlternatingLeastSquares

    os.makedirs(artifact_dir, exist_ok=True)

    if progress_callback:
        progress_callback("Menyiapkan confidence matrix...", 0.2)

    df = transaksi_df.copy()
    user_ids = df["user_id"].unique()
    game_names = df["game_name"].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    game_to_idx = {g: i for i, g in enumerate(game_names)}
    idx_to_game = {i: g for g, i in game_to_idx.items()}
    n_users, n_games = len(user_to_idx), len(game_to_idx)

    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["game_idx"] = df["game_name"].map(game_to_idx)

    # confidence: kalau ada rating & playtime_hours pakai itu, kalau tidak ada -> confidence seragam
    if "rating" in df.columns and "playtime_hours" in df.columns:
        rating_norm = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-9)
        playtime_log = np.log1p(df["playtime_hours"])
        playtime_norm = playtime_log / (playtime_log.max() + 1e-9)
        confidence = 1 + alpha_conf * (0.6 * rating_norm + 0.4 * playtime_norm)
    else:
        confidence = np.ones(len(df))

    ui_matrix = sp.csr_matrix((confidence, (df["user_idx"], df["game_idx"])), shape=(n_users, n_games))

    if progress_callback:
        progress_callback("Training ALS...", 0.5)

    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, random_state=42)
    model.fit(ui_matrix)

    if progress_callback:
        progress_callback("Menyimpan model...", 0.9)

    artifact = {
        "model": model, "user_item_matrix": ui_matrix,
        "user_to_idx": user_to_idx, "idx_to_game": idx_to_game,
    }
    with open(os.path.join(artifact_dir, "als_model.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    if progress_callback:
        progress_callback("ALS selesai.", 1.0)
    return artifact


# ============================================================================
# ORKESTRASI: jalankan ketiga training sekaligus
# ============================================================================
def train_all_models(transaksi_df: pd.DataFrame, dim_game_df: pd.DataFrame,
                      artifact_dir: str, progress_callback=None) -> dict:
    """Jalankan bundling -> forecast -> ALS secara berurutan, semua artifact
    ditulis ke SATU folder (artifact_dir) yang sama."""

    def _progress(stage_name, stage_weight_start, stage_weight_end):
        def cb(msg, pct):
            if progress_callback:
                overall = stage_weight_start + (stage_weight_end - stage_weight_start) * pct
                progress_callback(f"[{stage_name}] {msg}", overall)
        return cb

    os.makedirs(artifact_dir, exist_ok=True)
    # Simpan juga dim_game.csv apa adanya -> dibutuhkan inference layer untuk
    # XAI genre-overlap (explain_recommendation) saat memakai dataset custom.
    dim_game_df.to_csv(os.path.join(artifact_dir, "dim_game.csv"), index=False)

    results = {}
    results["bundling"] = train_bundling_model(
        transaksi_df, dim_game_df, artifact_dir, progress_callback=_progress("Bundling", 0.0, 0.3)
    )
    results["forecast"] = train_forecast_model(
        transaksi_df, dim_game_df, artifact_dir, progress_callback=_progress("Forecasting", 0.3, 0.7)
    )
    results["als"] = train_als_model(
        transaksi_df, artifact_dir, progress_callback=_progress("ALS", 0.7, 1.0)
    )
    return results
