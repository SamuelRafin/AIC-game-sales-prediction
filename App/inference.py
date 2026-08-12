"""
[3] INFERENCE LAYER
============================================================================
Modul ini menjembatani hasil training (artifact di [2]) dengan dashboard [4].
Semua fungsi di sini di-cache oleh Streamlit (@st.cache_resource) supaya
model/artifact cuma di-load SEKALI, bukan tiap kali user klik sesuatu.

3 fungsi utama (1 per model):
  - get_bundling_suggestions(game_name)      -> Cross-sell/Bundling
  - get_next_item_recommendations(user_id)   -> ALS Next-Item Recommendation
  - get_trending_genres() / get_genre_forecast(genre) -> Demand Forecasting

PENTING: sesuaikan PATH di bagian atas sesuai lokasi artifact di komputer kamu.
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------
# PATH ARTIFACT — SESUAIKAN DENGAN STRUKTUR FOLDER KAMU
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALS_MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "ALS", "als_model.pkl")
BUNDLING_PATH = os.path.join(BASE_DIR, "..", "artifacts", "association_table.pkl")
FORECAST_PATH = os.path.join(BASE_DIR, "..", "artifacts", "demand_forecast.pkl")


# ============================================================================
# 1. BUNDLING / CROSS-SELL
# ============================================================================
@st.cache_resource
def load_bundling_artifacts():
    with open(BUNDLING_PATH, "rb") as f:
        return pickle.load(f)


def get_bundling_suggestions(game_name: str, top_n: int = 5) -> dict:
    """Input: nama game (harus persis/exact match dari dataset).
    Output: dict berisi persentase kelakuan + daftar rekomendasi bundling."""
    data = load_bundling_artifacts()
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


def search_game_names(query: str, limit: int = 10) -> list:
    """Cari nama game yang mengandung 'query' (untuk autocomplete/dropdown di UI)."""
    data = load_bundling_artifacts()
    all_games = list(data["game_popularity"].keys())
    query_lower = query.lower()
    matches = [g for g in all_games if query_lower in g.lower()]
    return matches[:limit]


# ============================================================================
# 2. NEXT-ITEM RECOMMENDATION (ALS)
# ============================================================================
@st.cache_resource
def load_als_artifacts():
    with open(ALS_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _cosine_sim(vec_a, vec_b_matrix):
    """Cosine similarity antara 1 vektor vs banyak vektor sekaligus (untuk cari item paling mirip)."""
    norm_a = vec_a / (np.linalg.norm(vec_a) + 1e-9)
    norm_b = vec_b_matrix / (np.linalg.norm(vec_b_matrix, axis=1, keepdims=True) + 1e-9)
    return norm_b @ norm_a


def explain_recommendation(user_id: int, recommended_game_idx: int) -> dict:
    """
    XAI untuk ALS: model matrix factorization itu black-box (cuma dot product
    vektor laten, tidak ada alasan yang manusiawi secara default).
    Solusi: cari game DI HISTORI USER yang vektornya paling mirip (cosine similarity)
    dengan game yang direkomendasikan -> itu jadi "alasan" rekomendasinya.
    """
    artifacts = load_als_artifacts()
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
    most_similar_game = idx_to_game[owned_indices[best_idx_in_owned]]
    similarity_pct = round(float(sims[best_idx_in_owned]) * 100, 1)

    return {
        "alasan": f"Mirip dengan '{most_similar_game}' yang pernah kamu mainkan",
        "game_referensi": most_similar_game,
        "tingkat_kemiripan": similarity_pct,
    }


def get_next_item_recommendations(user_id: int, k: int = 10, with_explanation: bool = True) -> dict:
    """Input: user_id.
    Output: histori transaksi + rekomendasi game berikutnya beserta skor DAN alasan (XAI)."""
    artifacts = load_als_artifacts()
    model = artifacts["model"]
    user_item_matrix = artifacts["user_item_matrix"]
    user_to_idx = artifacts["user_to_idx"]
    idx_to_game = artifacts["idx_to_game"]

    if user_id not in user_to_idx:
        return {"error": f"User ID '{user_id}' tidak ditemukan di data."}

    uidx = user_to_idx[user_id]

    # Histori: game yang sudah pernah dibeli user ini (dari user_item_matrix)
    owned_indices = user_item_matrix[uidx].nonzero()[1]
    histori = [idx_to_game[i] for i in owned_indices]

    # Rekomendasi dari model ALS
    game_indices, scores = model.recommend(
        uidx, user_item_matrix[uidx], N=k, filter_already_liked_items=True
    )
    # Normalisasi skor mentah ALS -> persentase relatif (0-100%) supaya mudah dibaca di UI
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
def load_forecast_artifacts():
    try:
        with open(FORECAST_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Fallback to reading CSV artifacts if pickle has pandas version incompatibility
        artifacts_dir = os.path.dirname(FORECAST_PATH)
        sum_csv = os.path.join(artifacts_dir, "demand_trending_summary.csv")
        fc_csv = os.path.join(artifacts_dir, "demand_forecast.csv")
        
        trending_summary = pd.read_csv(sum_csv) if os.path.exists(sum_csv) else pd.DataFrame()
        forecast_df = pd.read_csv(fc_csv) if os.path.exists(fc_csv) else pd.DataFrame()
        
        forecast_by_genre = {}
        if not forecast_df.empty and "genre" in forecast_df.columns:
            for g, group in forecast_df.groupby("genre"):
                forecast_by_genre[g] = group.reset_index(drop=True)
                
        return {
            "trending_summary": trending_summary,
            "forecast_by_genre": forecast_by_genre
        }


def get_trending_genres(top_n: int = 5) -> dict:
    """Output: ranking genre paling trending naik & turun (untuk homepage)."""
    data = load_forecast_artifacts()
    trending = data["trending_summary"]
    return {
        "trending_naik": trending.head(top_n).to_dict("records"),
        "trending_turun": trending.tail(top_n).to_dict("records"),
    }


def get_genre_forecast(genre: str) -> pd.DataFrame:
    """Output: dataframe historis + forecast untuk 1 genre (untuk bikin grafik)."""
    data = load_forecast_artifacts()
    forecast_by_genre = data["forecast_by_genre"]
    if genre not in forecast_by_genre:
        return None
    return forecast_by_genre[genre]


def get_all_genres() -> list:
    data = load_forecast_artifacts()
    return sorted(data["forecast_by_genre"].keys())


def explain_forecast(genre: str) -> dict:
    """
    XAI untuk Demand Forecasting: Prophet secara native bisa dipecah jadi komponen
    trend (arah jangka panjang) vs yearly seasonality (pola musiman berulang tiap tahun).
    Ini menjelaskan APA yang mendorong forecast naik/turun, bukan cuma angka akhirnya.
    """
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
