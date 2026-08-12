"""
[2] MODEL TRAINING - Demand Forecasting (per Genre)
============================================================================
Prediksi demand (jumlah transaksi/pembelian) per genre untuk beberapa bulan
ke depan, pakai Prophet (Meta/Facebook).

Kenapa per GENRE (bukan per game individual)?
  - 10.205 game dengan total ~175rb transaksi -> rata-rata game cuma
    30-40 transaksi selama 4 tahun -> terlalu tipis untuk forecasting stabil.
  - Per genre jauh lebih padat datanya (tiap genre bisa dari ratusan game
    sekaligus), sehingga tren lebih reliable dibaca oleh model.

Kenapa Prophet?
  - Dirancang untuk business forecasting, robust untuk data historis pendek
    (kita cuma punya 48 bulan / 4 tahun data)
  - Otomatis menangani trend & seasonality, output termasuk confidence interval
  - Mudah dijelaskan & divisualisasikan untuk dashboard non-teknis

CATATAN PENTING: data transaksi bulanan di dataset ini relatif FLAT
(~3.500-3.900 transaksi/bulan, konsisten 2022-2025) - tidak ada trend/musiman
kuat. Jadi hasil forecast per genre juga akan cenderung stabil, bukan naik/
turun drastis - ini realistis sesuai pola data, bukan indikasi model salah.

Requirement: pip install prophet pandas numpy

Cara pakai:
    python train_demand_forecast.py
Output:
    ../artifacts/demand_forecast.csv     -> hasil forecast semua genre (untuk dashboard)
    ../artifacts/demand_forecast.pkl     -> dict lengkap + metadata trending
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import pickle
from prophet import Prophet

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # sesuaikan kalau struktur folder kamu beda
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

FORECAST_MONTHS = 6     # berapa bulan ke depan yang diprediksi
MIN_MONTHLY_AVG = 5     # genre dengan rata-rata transaksi/bulan di bawah ini di-skip (data terlalu tipis)


def main():
    t_start = time.time()
    print("=" * 70)
    print("TRAINING: Demand Forecasting per Genre (Prophet)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD & JOIN DATA
    # ------------------------------------------------------------------
    trans = pd.read_csv(TRANSAKSI_PATH)
    dim_game = pd.read_csv(DIM_GAME_PATH)
    trans["date_time"] = pd.to_datetime(trans["date_time"])
    trans["month"] = trans["date_time"].dt.to_period("M").dt.to_timestamp()

    dim_game_indexed = dim_game.drop_duplicates(subset="game_name").set_index("game_name")
    genre_lookup = dim_game_indexed[GENRE_COLS]

    trans = trans.merge(genre_lookup, left_on="game_name", right_index=True, how="left")
    trans[GENRE_COLS] = trans[GENRE_COLS].fillna(0)

    print(f"Transaksi: {len(trans):,} baris, {trans['month'].nunique()} bulan unik")

    # ------------------------------------------------------------------
    # 2. AGREGASI: jumlah transaksi per genre per bulan
    #    (1 transaksi bisa masuk ke beberapa genre kalau game-nya multi-genre)
    # ------------------------------------------------------------------
    print("\nMenghitung demand bulanan per genre...")
    genre_monthly = {}
    for genre in GENRE_COLS:
        sub = trans[trans[genre] == 1]
        if len(sub) == 0:
            continue
        monthly_counts = sub.groupby("month").size().reset_index(name="y")
        monthly_counts.rename(columns={"month": "ds"}, inplace=True)
        genre_monthly[genre] = monthly_counts

    # ------------------------------------------------------------------
    # 3. TRAIN PROPHET PER GENRE + FORECAST
    # ------------------------------------------------------------------
    print(f"\nTraining Prophet untuk {len(genre_monthly)} genre...")
    all_forecasts = []
    trending_summary = []
    skipped = []

    for genre, df_genre in genre_monthly.items():
        avg_monthly = df_genre["y"].mean()
        if avg_monthly < MIN_MONTHLY_AVG or len(df_genre) < 12:
            skipped.append(genre)
            continue

        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(df_genre)

        future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq="MS")
        forecast = model.predict(future)

        # Ambil juga komponen trend & yearly seasonality -> untuk XAI (jelaskan KENAPA forecast naik/turun)
        component_cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        if "yearly" in forecast.columns:
            component_cols.append("yearly")
        forecast_out = forecast[component_cols].copy()
        if "yearly" not in forecast_out.columns:
            forecast_out["yearly"] = 0.0
        forecast_out["genre"] = genre
        forecast_out["is_forecast"] = forecast_out["ds"] > df_genre["ds"].max()
        forecast_out = forecast_out.merge(df_genre[["ds", "y"]], on="ds", how="left")
        all_forecasts.append(forecast_out)

        # Ringkasan trending: bandingkan rata-rata forecast vs rata-rata actual 3 bulan terakhir
        actual_recent = df_genre.sort_values("ds")["y"].tail(3).mean()
        forecast_future = forecast_out[forecast_out["is_forecast"]]["yhat"].mean()
        pct_change = ((forecast_future - actual_recent) / actual_recent) * 100 if actual_recent > 0 else 0

        trending_summary.append({
            "genre": genre,
            "rata2_actual_3bulan_terakhir": round(actual_recent, 1),
            "rata2_forecast_kedepan": round(forecast_future, 1),
            "persen_perubahan": round(pct_change, 2),
            "total_historis": int(df_genre["y"].sum()),
        })

    print(f"Genre yang di-skip (data terlalu tipis): {skipped}")

    # ------------------------------------------------------------------
    # 4. SUSUN & SIMPAN ARTIFACT
    # ------------------------------------------------------------------
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    trending_df = pd.DataFrame(trending_summary).sort_values("persen_perubahan", ascending=False)

    csv_path = os.path.join(ARTIFACT_DIR, "demand_forecast.csv")
    forecast_df.to_csv(csv_path, index=False)
    print(f"\nDisimpan: {csv_path} ({len(forecast_df):,} baris)")

    trending_csv_path = os.path.join(ARTIFACT_DIR, "demand_trending_summary.csv")
    trending_df.to_csv(trending_csv_path, index=False)
    print(f"Disimpan: {trending_csv_path}")

    pkl_path = os.path.join(ARTIFACT_DIR, "demand_forecast.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "forecast_by_genre": {g: df for g, df in forecast_df.groupby("genre")},
            "trending_summary": trending_df,
            "forecast_months": FORECAST_MONTHS,
            "skipped_genres": skipped,
        }, f)
    print(f"Disimpan: {pkl_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal waktu training: {elapsed:.1f} detik")

    # ------------------------------------------------------------------
    # 5. RINGKASAN UNTUK HOMEPAGE ("genre yang bakal trending ke depan")
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP 5 GENRE TRENDING NAIK (untuk homepage dashboard)")
    print("=" * 70)
    print(trending_df.head(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 5 GENRE TRENDING TURUN")
    print("=" * 70)
    print(trending_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
