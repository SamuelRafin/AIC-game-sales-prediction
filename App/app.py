"""
[4] DASHBOARD (Streamlit)
============================================================================
Solusi untuk 3 model dengan jenis input berbeda: SATU dashboard, dengan
MODE SELECTOR di awal. User pilih mode dulu -> input yang muncul menyesuaikan
-> panggil fungsi inference yang sesuai.

  Mode "Beranda"      -> tanpa input, tampilkan Demand Forecasting (trending)
  Mode "Cari Game"    -> input: nama game -> panggil Bundling
  Mode "Cari User ID" -> input: user_id   -> panggil ALS Recommendation

Requirement: pip install streamlit pandas numpy implicit plotly

Cara jalankan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from inference import (
    get_bundling_suggestions, search_game_names,
    get_next_item_recommendations,
    get_trending_genres, get_genre_forecast, get_all_genres, explain_forecast,
)

st.set_page_config(page_title="Game Store Analytics", page_icon="🎮", layout="wide")

# ------------------------------------------------------------------
# HEADER + MODE SELECTOR (INI SOLUSI UNTUK "BAGAIMANA HANDLE INPUT"-NYA)
# ------------------------------------------------------------------
st.title("🎮 Game Store Analytics Dashboard")

mode = st.radio(
    "Pilih mode:",
    ["🏠 Beranda (Trending)", "🔍 Cari berdasarkan Game (Bundling)", "👤 Cari berdasarkan User ID (Rekomendasi)"],
    horizontal=True,
)

st.divider()

# ============================================================================
# MODE 1: BERANDA — Demand Forecasting (tanpa input wajib)
# ============================================================================
if mode == "🏠 Beranda (Trending)":
    st.subheader("📈 Genre yang Diprediksi Trending")

    trending = get_trending_genres(top_n=5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔺 Trending Naik**")
        df_naik = pd.DataFrame(trending["trending_naik"])
        st.dataframe(df_naik[["genre", "persen_perubahan", "rata2_forecast_kedepan"]], hide_index=True)
    with col2:
        st.markdown("**🔻 Trending Turun**")
        df_turun = pd.DataFrame(trending["trending_turun"])
        st.dataframe(df_turun[["genre", "persen_perubahan", "rata2_forecast_kedepan"]], hide_index=True)

    st.divider()

    # Input OPSIONAL di dalam mode ini: pilih genre spesifik untuk lihat grafik detail
    st.markdown("**Lihat detail forecast per genre:**")
    all_genres = get_all_genres()
    selected_genre = st.selectbox("Pilih genre:", all_genres)

    if selected_genre:
        forecast_df = get_genre_forecast(selected_genre)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"].round(0), name="Forecast", line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"].round(0), name="Upper", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["yhat_lower"].round(0), name="Confidence Interval",
            fill="tonexty", line=dict(width=0), fillcolor="rgba(65,105,225,0.15)",
        ))
        fig.update_layout(title=f"Demand Forecast: {selected_genre}", xaxis_title="Bulan", yaxis_title="Jumlah Transaksi", yaxis=dict(tickformat=",d"))
        st.plotly_chart(fig, use_container_width=True)

        # XAI: jelaskan APA yang mendorong forecast ini (trend vs musiman)
        with st.expander("💡 Kenapa forecast-nya begini?"):
            penjelasan = explain_forecast(selected_genre)
            st.write(penjelasan["penjelasan"])
            colA, colB = st.columns(2)
            colA.metric("Perubahan Trend", f"{penjelasan['perubahan_trend_persen']:+.1f}%")
            colB.metric("Efek Musiman Rata-rata", f"{penjelasan['rata2_efek_musiman']:+.1f} transaksi/bulan")

# ============================================================================
# MODE 2: CARI GAME — Bundling / Cross-sell
# ============================================================================
elif mode == "🔍 Cari berdasarkan Game (Bundling)":
    st.subheader("🔍 Cari Saran Bundling per Game")

    # INPUT: search box dengan autocomplete (bukan dropdown 10rb+ item -> lambat & tidak praktis)
    query = st.text_input("Ketik nama game:", placeholder="Contoh: Valorant")

    selected_game = None
    if query:
        matches = search_game_names(query, limit=10)
        if matches:
            selected_game = st.selectbox("Pilih game yang dimaksud:", matches)
        else:
            st.warning("Game tidak ditemukan. Coba kata kunci lain.")

    if selected_game:
        hasil = get_bundling_suggestions(selected_game, top_n=5)

        if "error" in hasil:
            st.error(hasil["error"])
        else:
            st.metric(
                label=f"Persentase Kelakuan: {selected_game}",
                value=f"{hasil['persen_laku']}%",
                help=f"{hasil['jumlah_pembeli']} dari total user membeli game ini",
            )

            st.markdown("**Saran Bundling (genre sama, sering dibeli bersamaan):**")
            if hasil["rekomendasi_bundling"]:
                for item in hasil["rekomendasi_bundling"]:
                    st.progress(
                        min(item["persen_bundling"] / 100, 1.0),
                        text=f"{item['game_B']} — {item['persen_bundling']}% ({item['jumlah_beli_keduanya']} user beli keduanya)",
                    )
            else:
                st.info("Belum ada data bundling yang cukup untuk game ini.")

# ============================================================================
# MODE 3: CARI USER ID — Next-Item Recommendation (ALS)
# ============================================================================
elif mode == "👤 Cari berdasarkan User ID (Rekomendasi)":
    st.subheader("👤 Rekomendasi Personal per User")

    # INPUT: number input untuk user_id
    user_id_input = st.number_input("Masukkan User ID:", min_value=1, step=1, value=None, placeholder="Contoh: 123")

    if user_id_input:
        hasil = get_next_item_recommendations(int(user_id_input), k=10)

        if "error" in hasil:
            st.error(hasil["error"])
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**Histori Transaksi** ({hasil['jumlah_histori']} game)")
                st.dataframe(pd.DataFrame({"game": hasil["histori_transaksi"]}), hide_index=True, height=300)

            with col2:
                st.markdown("**Rekomendasi Game Berikutnya**")
                for item in hasil["rekomendasi"]:
                    st.progress(
                        min(item["skor_kemiripan"] / 100, 1.0),
                        text=f"{item['game']} — skor kecocokan {item['skor_kemiripan']}%",
                    )
                    # XAI: jelaskan kenapa game ini direkomendasikan
                    if "xai" in item:
                        st.caption(f"   💡 {item['xai']['alasan']} (kemiripan {item['xai']['tingkat_kemiripan']}%)")
