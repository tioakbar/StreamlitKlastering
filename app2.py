import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import plotly.express as px
import plotly.graph_objects as go

# ====================================================
# KONFIGURASI & STYLING GLOBAL
# ====================================================
st.set_page_config(
    page_title="Dashboard Clustering PMA Surabaya",
    page_icon="💹",
    layout="wide"
)

# CSS untuk header, tab, dan animasi fade-in
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f172a, #1d4ed8);
        padding: 20px 25px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 28px;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 14px;
        margin-top: 0;
        opacity: 0.9;
    }
    .small-note {
        font-size: 12px;
        color: #6b7280;
    }

    /* ===== Styling & Animasi TAB ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background-color: #f3f4f6;
        color: #374151;
        font-weight: 500;
        border: none;
        transition:
            background-color 0.25s ease,
            color 0.25s ease,
            transform 0.20s ease,
            box-shadow 0.20s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15,23,42,0.18);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg,#1d4ed8,#6366f1);
        color: #ffffff;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 10px 24px rgba(15,23,42,0.30);
    }

    /* ===== Animasi fade-in isi tab ===== */
    .fade-in {
        animation: fadeIn 0.35s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Highlight kartu hasil prediksi */
    .prediction-card {
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(120deg,#22c55e1a,#22c55e33);
        border: 1px solid #22c55e55;
        box-shadow: 0 12px 30px rgba(22,163,74,0.28);
        margin-bottom: 12px;
    }
    .prediction-card h3 {
        margin: 0 0 6px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ====================================================
# HELPER FUNCTIONS
# ====================================================
def winsorize_like_r(df, lower_q=0.05, upper_q=0.95):
    """
    Meniru logika winsorizing di skrip R:
    - hitung Q1, Q3, IQR
    - nilai di bawah (Q1 - 1.5*IQR) diganti quantile lower_q
    - nilai di atas (Q3 + 1.5*IQR) diganti quantile upper_q
    """
    df = df.copy()
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        x = df[col].to_numpy(dtype=float)
        q1, q3 = np.quantile(x, [0.25, 0.75])
        caps = np.quantile(x, [lower_q, upper_q])
        H = 1.5 * (q3 - q1)

        lower_bound = q1 - H
        upper_bound = q3 + H

        x = np.where(x < lower_bound, caps[0], x)
        x = np.where(x > upper_bound, caps[1], x)

        df[col] = x

    return df


def compute_elbow_silhouette(X, k_min=2, k_max=10, random_state=42):
    """
    Menghitung SSE (Elbow) dan Silhouette Score untuk rentang K.
    """
    sse = []
    Ks = list(range(1, k_max + 1))
    sil_scores = []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sse.append(km.inertia_)

        if k >= k_min:
            sil = silhouette_score(X, labels)
            sil_scores.append((k, sil))

    return Ks, sse, sil_scores


def add_cluster_to_df(df_raw, labels, pca_coords, negara_col="Negara"):
    """
    Menggabungkan label cluster + koordinat PCA ke data asli.
    """
    df_result = df_raw.copy()
    df_result["Cluster"] = labels

    pca_df = pd.DataFrame(
        pca_coords,
        columns=["PC1", "PC2"]
    )

    df_result = pd.concat([df_result, pca_df], axis=1)

    if negara_col in df_result.columns:
        cols = ["Cluster", "PC1", "PC2"] + [c for c in df_result.columns if c not in ["Cluster", "PC1", "PC2"]]
        df_result = df_result[cols]

    return df_result


def download_csv_button(df, filename="hasil_cluster.csv", label="📥 Download Hasil Clustering (.csv)"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def generate_cluster_text_summary(df_profile, selected_features):
    """
    Buat teks penjelasan otomatis untuk profil cluster.
    """
    lines = []
    for _, row in df_profile.iterrows():
        c = row["Cluster"]
        desc_parts = []
        for feat in selected_features:
            value = row[feat]
            desc_parts.append(f"{feat} ≈ {value:,.2f}")
        desc = "; ".join(desc_parts)
        lines.append(f"- **Cluster {c}**: {desc}")
    return "\n".join(lines)


# ====================================================
# HEADER UTAMA
# ====================================================
st.markdown("""
<div class="main-header">
  <h1>💹 Dashboard Clustering Penanaman Modal Asing (PMA) – Kota Surabaya</h1>
  <p>Pipeline: Winsorizing → Z-Score → PCA → Penentuan K → K-Means → Evaluasi Cluster & Prediksi</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='small-note'>File <b>PMA_Investasi2 (1).xlsx</b> dibaca langsung dari folder yang sama dengan aplikasi ini. "
    "Gunakan tab di bawah untuk menjelajahi analisis.</p>",
    unsafe_allow_html=True
)

# ====================================================
# SIDEBAR
# ====================================================
st.sidebar.title("⚙️ Pengaturan")

st.sidebar.markdown(
    "Pastikan file **`PMA_Investasi2 (1).xlsx`** berada di folder yang sama dengan `app.py`."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Preprocessing")

winsor_lower = st.sidebar.slider("Quantile Winsor Bawah", 0.0, 0.2, 0.05, 0.01)
winsor_upper = st.sidebar.slider("Quantile Winsor Atas", 0.8, 1.0, 0.95, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 K-Means & PCA")

k_max_elbow = st.sidebar.slider("Maksimum K (Elbow & Silhouette)", 4, 12, 8, 1)
k_selected = st.sidebar.slider("Jumlah Cluster K-Means (K)", 2, 8, 2, 1)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.info("Untuk meniru hasil jurnal & skrip R, gunakan K = 2.")


# ====================================================
# LOAD DATA DARI FILE LOKAL
# ====================================================
DATA_PATH = "PMA_Investasi2 (1).xlsx"

try:
    with st.spinner(f"📥 Membaca dataset dari `{DATA_PATH}` ..."):
        df_raw = pd.read_excel(DATA_PATH)
    st.success(f"Dataset berhasil dimuat. Baris: {df_raw.shape[0]}, Kolom: {df_raw.shape[1]}")
except FileNotFoundError:
    st.error(
        f"File `{DATA_PATH}` tidak ditemukan.\n\n"
        "Taruh file Excel tersebut di folder yang sama dengan `app.py`, lalu jalankan ulang aplikasi."
    )
    st.stop()

# Deteksi nama kolom negara & fitur numerik
negara_col = None
for candidate in ["Negara", "Country", "NEGARA"]:
    if candidate in df_raw.columns:
        negara_col = candidate
        break

numeric_cols = [c for c in df_raw.columns if np.issubdtype(df_raw[c].dtype, np.number)]
default_features = [c for c in numeric_cols]

st.subheader("📦 Cuplikan Data Asli")
st.dataframe(df_raw.head(), use_container_width=True)

st.markdown("### 🔧 Pilih Fitur Numerik untuk Clustering")
selected_features = st.multiselect(
    "Minimal 2 fitur:",
    options=numeric_cols,
    default=default_features,
    help="Biasanya: Nilai Investasi, Jumlah Proyek, TKI, TKA"
)

if len(selected_features) < 2:
    st.error("Minimal pilih 2 fitur numerik untuk PCA & K-Means.")
    st.stop()

data_selected = df_raw[selected_features].copy()

# ====================================================
# TABS UTAMA: Dashboard + Analisis
# ====================================================
tab_home, tab1, tab2, tab3, tab_pred, tab4 = st.tabs([
    "🏠 Dashboard Utama",
    "📊 EDA & Preprocessing",
    "🧬 PCA & Cari K Terbaik",
    "🧩 Clustering & Insight",
    "🤖 Prediksi Cluster",
    "📥 Download & Ringkasan"
])

# ====================================================
# TAB HOME – DASHBOARD UTAMA
# ====================================================
with tab_home:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("🏠 Dashboard Utama")

    st.markdown("""
    Halaman ini memberikan gambaran umum tentang **PMA Kota Surabaya** sebelum masuk ke analisis yang lebih teknis.
    Cocok ditampilkan saat presentasi atau sebagai landing page dashboard.
    """)

    # Deteksi kolom penting secara fleksibel
    invest_col = next((c for c in df_raw.columns if "invest" in c.lower()), None)
    proj_col   = next((c for c in df_raw.columns if "proyek" in c.lower() or "project" in c.lower()), None)
    tki_col    = next((c for c in df_raw.columns if "tki" in c.lower()), None)
    tka_col    = next((c for c in df_raw.columns if "tka" in c.lower()), None)

    # ---- KPI Cards ----
    colA, colB, colC, colD = st.columns(4)

    with colA:
        st.metric("Jumlah Negara Investor", df_raw.shape[0])

    if invest_col:
        total_inv = df_raw[invest_col].sum()
        avg_inv   = df_raw[invest_col].mean()
        with colB:
            st.metric("Total Nilai Investasi", f"{total_inv:,.0f}")
        with colC:
            st.metric("Rata-rata Investasi/Negara", f"{avg_inv:,.0f}")
    else:
        with colB:
            st.metric("Total Nilai Investasi", "-")
        with colC:
            st.metric("Rata-rata Investasi/Negara", "-")

    if proj_col:
        total_proj = df_raw[proj_col].sum()
        with colD:
            st.metric("Total Jumlah Proyek", f"{int(total_proj):,}")
    else:
        with colD:
            st.metric("Total Jumlah Proyek", "-")

    st.markdown("""
    **Interpretasi cepat:**
    - Jumlah negara investor menunjukkan seberapa beragam sumber PMA di Surabaya.
    - Total dan rata-rata investasi menggambarkan skala ekonomi yang terlibat.
    - Total proyek menunjukkan seberapa aktif realisasi investasi yang berjalan.
    """)

    # ---- Top 10 Negara berdasarkan Investasi ----
    if invest_col and negara_col:
        st.subheader("🏆 Top 10 Negara Berdasarkan Nilai Investasi")

        top10 = (
            df_raw[[negara_col, invest_col]]
            .sort_values(invest_col, ascending=False)
            .head(10)
        )

        fig_top = px.bar(
            top10,
            x=negara_col,
            y=invest_col,
            title="Top 10 Negara Investor PMA (Berdasarkan Nilai Investasi)",
            text=invest_col
        )
        fig_top.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_top.update_layout(
            xaxis_title="Negara",
            yaxis_title="Nilai Investasi",
            xaxis_tickangle=-45,
            transition_duration=500
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("""
        **Cara baca grafik:**
        - Batang paling tinggi menunjukkan negara dengan nilai investasi terbesar.
        - Posisi batang menggambarkan perbandingan antarnegara.
        - Bisa dipakai untuk menjawab: *“Negara mana yang paling dominan dalam PMA Surabaya?”*
        """)

        # ---- Pie chart share investasi Top 10 ----
        st.subheader("🧩 Pangsa Investasi (Top 10 Negara)")

        fig_pie = px.pie(
            top10,
            values=invest_col,
            names=negara_col,
            hole=0.4,
            title="Share Nilai Investasi PMA – Top 10 Negara"
        )
        fig_pie.update_layout(transition_duration=500)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
        **Interpretasi:**
        - Diagram donat ini menunjukkan persentase kontribusi masing-masing negara terhadap total investasi 10 negara teratas.
        - Semakin besar sektor donat, semakin besar peran negara tersebut dalam PMA.
        """)
    else:
        st.info("Kolom investasi atau nama negara tidak terdeteksi. Sesuaikan nama kolom di dataset jika perlu.")

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# TAB 1 – EDA & PREPROCESSING
# ====================================================
with tab1:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("1️⃣ EDA & Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Ringkasan Statistik (Sebelum Winsorizing)")
        desc_before = data_selected.describe().T
        st.dataframe(desc_before, use_container_width=True)

    with col2:
        st.subheader("📉 Boxplot Sebelum Winsorizing")
        df_melt_before = data_selected.melt(var_name="Fitur", value_name="Nilai")
        fig_box_before = px.box(
            df_melt_before,
            x="Fitur",
            y="Nilai",
            title="Distribusi Fitur Sebelum Winsorizing"
        )
        fig_box_before.update_layout(xaxis_tickangle=-45, transition_duration=500)
        st.plotly_chart(fig_box_before, use_container_width=True)

    st.markdown("""
    **Penjelasan:**
    - Ringkasan statistik menunjukkan sebaran nilai tiap fitur (min, max, mean, dll).
    - Boxplot membantu mendeteksi outlier, misalnya negara dengan investasi jauh lebih tinggi dari yang lain.
    """)

    # Winsorizing
    with st.spinner("✂️ Melakukan Winsorizing untuk mengendalikan outlier..."):
        data_winsor = winsorize_like_r(data_selected, lower_q=winsor_lower, upper_q=winsor_upper)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📌 Ringkasan Statistik (Setelah Winsorizing)")
        desc_after = data_winsor.describe().T
        st.dataframe(desc_after, use_container_width=True)

    with col4:
        st.subheader("📉 Boxplot Setelah Winsorizing")
        df_melt_after = data_winsor.melt(var_name="Fitur", value_name="Nilai")
        fig_box_after = px.box(
            df_melt_after,
            x="Fitur",
            y="Nilai",
            title="Distribusi Fitur Setelah Winsorizing"
        )
        fig_box_after.update_layout(xaxis_tickangle=-45, transition_duration=500)
        st.plotly_chart(fig_box_after, use_container_width=True)

    # Normalisasi Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_winsor)
    data_scaled = pd.DataFrame(X_scaled, columns=selected_features)

    st.subheader("📉 Boxplot Setelah Normalisasi (Z-Score)")
    df_melt_scaled = data_scaled.melt(var_name="Fitur", value_name="Nilai Z")
    fig_box_scaled = px.box(
        df_melt_scaled,
        x="Fitur",
        y="Nilai Z",
        title="Distribusi Fitur Setelah Normalisasi"
    )
    fig_box_scaled.update_layout(xaxis_tickangle=-45, transition_duration=500)
    st.plotly_chart(fig_box_scaled, use_container_width=True)

    st.subheader("🧱 Heatmap Korelasi (Setelah Normalisasi)")
    corr = data_scaled.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Korelasi Antar Fitur"
    )
    fig_corr.update_layout(transition_duration=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Penjelasan:**
    - Winsorizing membuat data ekstrem tidak terlalu mendominasi.
    - Normalisasi Z-Score menyamakan skala fitur sehingga K-Means tidak bias.
    - Heatmap korelasi menunjukkan hubungan antar fitur; korelasi tinggi mendukung penggunaan PCA.
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# TAB 2 – PCA & CARI K TERBAIK
# ====================================================
with tab2:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("2️⃣ PCA & Penentuan Jumlah Cluster (K)")

    with st.spinner("🧬 Menghitung PCA..."):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

    st.subheader("📈 Rasio Varian yang Dijelaskan PCA")
    st.write(
        f"PC1: **{explained_var[0]:.3f}**, "
        f"PC2: **{explained_var[1]:.3f}**, "
        f"Total: **{explained_var.sum():.3f}**"
    )

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    if negara_col is not None:
        pca_df[negara_col] = df_raw[negara_col]

    st.subheader("🌐 Peta PCA (Tanpa Cluster)")
    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_data=[negara_col] if negara_col is not None else None,
        title="Proyeksi Data ke Ruang PCA (2 Komponen Utama)",
    )
    fig_pca.update_layout(transition_duration=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("""
    **Penjelasan:**
    - PCA mereduksi dimensi menjadi dua komponen utama sambil mempertahankan sebagian besar informasi.
    - Titik yang berdekatan menunjukkan negara dengan pola PMA yang mirip.
    """)

    st.subheader("📉 Elbow Method (SSE vs K)")
    with st.spinner("📊 Menghitung SSE & Silhouette..."):
        Ks, sse, sil_scores = compute_elbow_silhouette(
            X_pca,
            k_min=2,
            k_max=k_max_elbow,
            random_state=random_state
        )

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=Ks,
        y=sse,
        mode="lines+markers",
        name="SSE"
    ))
    fig_elbow.update_layout(
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="SSE (Within-Cluster Sum of Squares)",
        title="Elbow Method",
        hovermode="x unified",
        transition_duration=500
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.subheader("📈 Silhouette Score vs K")
    if sil_scores:
        ks_sil, vals_sil = zip(*sil_scores)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=ks_sil,
            y=vals_sil,
            mode="lines+markers",
            name="Silhouette"
        ))
        fig_sil.update_layout(
            xaxis_title="Jumlah Cluster (K)",
            yaxis_title="Silhouette Score",
            title="Silhouette Score untuk Berbagai K",
            hovermode="x unified",
            transition_duration=500
        )
        st.plotly_chart(fig_sil, use_container_width=True)

        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
        st.info(
            f"Silhouette tertinggi diperoleh pada K = **{best_k}** "
            f"dengan nilai **{best_sil:.3f}**. "
            f"Ini bisa jadi rujukan pemilihan K di sidebar."
        )
    else:
        st.warning("Silhouette hanya dihitung mulai K=2. Coba naikkan parameter K maksimum di sidebar.")

    st.markdown("""
    **Penjelasan:**
    - Elbow Method mencari titik 'siku' sebagai indikasi K optimal.
    - Silhouette Score mengukur seberapa baik pemisahan cluster (semakin tinggi semakin baik).
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# TAB 3 – CLUSTERING & INSIGHT
# ====================================================
with tab3:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("3️⃣ K-Means Clustering & Insight")

    with st.spinner("🚀 Menjalankan K-Means..."):
        km = KMeans(n_clusters=k_selected, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        dbi = davies_bouldin_score(X_pca, labels)

        df_clustered = add_cluster_to_df(df_raw, labels, X_pca, negara_col=negara_col)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Jumlah Cluster (K)", k_selected)
    col_m2.metric("Silhouette Score", f"{sil:.3f}")
    col_m3.metric("Davies-Bouldin Index", f"{dbi:.3f}")

    st.markdown("""
    **Penjelasan metrik:**
    - **Silhouette Score** mendekati 1 → cluster rapi & terpisah; mendekati 0 → cluster tumpang tindih.
    - **Davies-Bouldin Index (DBI)** semakin kecil → cluster semakin baik.
    """)

    st.subheader("🌐 Peta PCA dengan Label Cluster")
    plot_df = df_clustered.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    hover_cols = ["Cluster"]
    if negara_col is not None:
        hover_cols.append(negara_col)

    fig_cluster = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_cols,
        title=f"Pemetaan PCA dengan K-Means (K = {k_selected})",
        symbol="Cluster"
    )
    fig_cluster.update_traces(marker=dict(size=10, line=dict(width=1)))
    fig_cluster.update_layout(transition_duration=500)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("""
    **Cara baca grafik:**
    - Setiap titik = satu negara investor.
    - Warna = cluster hasil K-Means.
    - Kelompok titik dengan warna yang sama dan letak berdekatan → negara dengan karakteristik PMA yang mirip.
    """)

    st.subheader("🎞️ Animasi: Negara per Cluster di Ruang PCA")
    fig_cluster_anim = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_cols,
        animation_frame="Cluster",
        title="Animasi: Visualisasi Negara per Cluster di Ruang PCA"
    )
    fig_cluster_anim.update_traces(marker=dict(size=10, line=dict(width=1)))
    st.plotly_chart(fig_cluster_anim, use_container_width=True)

    st.subheader("📃 Tabel Hasil Clustering per Negara")
    st.dataframe(df_clustered, use_container_width=True)

    st.markdown("""
    **Penjelasan tabel:**
    - `Cluster` menunjukkan cluster hasil K-Means.
    - `PC1` & `PC2` adalah koordinat PCA untuk visualisasi.
    - Kolom lain adalah variabel asli (investasi, proyek, TKI, TKA, dll).
    """)

    st.subheader("📊 Profil Rata-Rata Tiap Cluster (di Ruang Asli Fitur)")
    df_profile = df_clustered.groupby("Cluster")[selected_features].mean().reset_index()
    df_profile["Cluster"] = df_profile["Cluster"].astype(str)

    long_profile = df_profile.melt(id_vars="Cluster", var_name="Fitur", value_name="Rata-Rata")

    fig_prof_anim = px.bar(
        long_profile,
        x="Fitur",
        y="Rata-Rata",
        color="Cluster",
        animation_frame="Cluster",
        range_y=[0, long_profile["Rata-Rata"].max() * 1.1],
        title="Animasi Profil Rata-Rata Fitur per Cluster"
    )
    fig_prof_anim.update_layout(xaxis_tickangle=-45, transition_duration=500)
    st.plotly_chart(fig_prof_anim, use_container_width=True)

    fig_prof = px.bar(
        long_profile,
        x="Fitur",
        y="Rata-Rata",
        color="Cluster",
        barmode="group",
        title="Perbandingan Rata-Rata Fitur per Cluster (Statik)",
    )
    fig_prof.update_layout(xaxis_tickangle=-45, transition_duration=500)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.markdown("**Ringkasan otomatis profil cluster:**")
    summary_text = generate_cluster_text_summary(df_profile, selected_features)
    st.markdown(summary_text)

    st.markdown("""
    **Cara baca profil cluster:**
    - Cluster dengan nilai rata-rata `Nilai Investasi` & `Jumlah Proyek` tinggi → kelompok negara kontributor utama PMA.
    - Jika `TKI` tinggi dan `TKA` rendah → investasi cenderung menyerap tenaga kerja lokal.
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# TAB PREDIKSI – PREDIKSI CLUSTER UNTUK DATA BARU
# ====================================================
with tab_pred:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("🤖 Prediksi Cluster untuk Negara / Skenario Baru")

    st.markdown("""
    Masukkan nilai fitur (misalnya skenario negara baru atau perubahan investasi), lalu tekan tombol **Prediksi Cluster**.
    
    Sistem akan:
    1. Menjalankan pipeline yang sama (Winsorizing → Z-Score → PCA → K-Means).
    2. Menentukan data tersebut masuk **cluster ke berapa**.
    3. Menampilkan **animasi** posisi titik baru di ruang PCA.
    4. Menampilkan **party animation + sound** ketika prediksi berhasil.
    5. Memberikan **interpretasi otomatis** berdasarkan profil cluster.
    """)

    with st.form("form_prediksi"):
        cols = st.columns(2)
        input_values = {}

        for i, feat in enumerate(selected_features):
            col = cols[i % 2]
            min_v = float(df_raw[feat].min())
            max_v = float(df_raw[feat].max())
            default = float(df_raw[feat].mean())
            with col:
                val = st.number_input(
                    feat,
                    value=default,
                    min_value=min_v,
                    max_value=max_v
                )
            input_values[feat] = val

        submitted = st.form_submit_button("🔮 Prediksi Cluster")

    if submitted:
        # 🎉 PARTY MODE: balon + sound
        st.balloons()
        st.markdown("""
            <audio autoplay>
                <source src="https://www.soundjay.com/human/sounds/cheering-6.mp3" type="audio/mpeg">
            </audio>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='prediction-card'>"
            "<h3>🎊 Prediksi Berhasil!</h3>"
            "<p>Berikut hasil pengelompokan dan interpretasi untuk skenario yang kamu masukkan.</p>"
            "</div>",
            unsafe_allow_html=True
        )

        st.subheader("📌 Hasil Prediksi")

        # Data baru
        df_input = pd.DataFrame([input_values], columns=selected_features)

        # Pakai scaler & PCA yang sudah dilatih
        X_scaled_new = scaler.transform(df_input)
        X_pca_new = pca.transform(X_scaled_new)
        cluster_pred = int(km.predict(X_pca_new)[0])

        col_pred1, col_pred2 = st.columns([1, 1])

        with col_pred1:
            st.success(f"Data baru diprediksi masuk **Cluster {cluster_pred}**.")

            # Siapkan data untuk animasi PCA
            pca_new_df = pd.DataFrame(X_pca_new, columns=["PC1", "PC2"])
            pca_new_df["Cluster"] = str(cluster_pred)
            pca_new_df["Jenis"] = "Data Baru"

            pca_old_df = plot_df.copy()
            pca_old_df["Jenis"] = "Data Lama"

            comb = pd.concat([
                pca_old_df[["PC1", "PC2", "Cluster", "Jenis"]],
                pca_new_df[["PC1", "PC2", "Cluster", "Jenis"]]
            ])

            # Animasi: frame "Data Lama" -> "Data Baru"
            fig_pred = px.scatter(
                comb,
                x="PC1",
                y="PC2",
                color="Cluster",
                hover_data=["Jenis"],
                animation_frame="Jenis",
                title="🎞️ Animasi Posisi Data Baru di Ruang PCA"
            )
            fig_pred.update_traces(marker=dict(size=10, line=dict(width=1)))
            fig_pred.update_layout(transition_duration=600)
            st.plotly_chart(fig_pred, use_container_width=True)

        with col_pred2:
            # Hitung jarak ke centroid tiap cluster
            centroids = km.cluster_centers_
            dists = np.linalg.norm(centroids - X_pca_new[0], axis=1)
            dist_df = pd.DataFrame({
                "Cluster": list(range(k_selected)),
                "Jarak ke Centroid": dists
            })

            st.markdown("**📏 Jarak ke centroid tiap cluster:**")
            st.dataframe(dist_df.style.format({"Jarak ke Centroid": "{:.3f}"}), use_container_width=True)

            fig_dist = px.bar(
                dist_df,
                x="Cluster",
                y="Jarak ke Centroid",
                title="📊 Jarak Data Baru ke Masing-Masing Centroid",
            )
            fig_dist.update_layout(transition_duration=400)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Skor "confidence" sederhana berbasis jarak centroid
            # (semakin dekat ke centroid cluster_pred → confidence makin tinggi)
            max_dist = dists.max() if dists.max() > 0 else 1.0
            confidence = 1 - (dists[cluster_pred] / max_dist)
            st.metric("Confidence (relatif)", f"{confidence*100:,.1f} %")

        # Interpretasi berbasis profil cluster
        st.subheader("🧠 Interpretasi Otomatis")

        df_profile_pred = df_clustered.groupby("Cluster")[selected_features].mean().reset_index()
        df_profile_pred["Cluster"] = df_profile_pred["Cluster"].astype(int)

        # Tentukan fitur utama untuk ranking (pakai investasi jika ada)
        if invest_col and invest_col in selected_features:
            main_feat = invest_col
        else:
            main_feat = selected_features[0]

        rank_series = (
            df_profile_pred
            .set_index("Cluster")[main_feat]
            .sort_values(ascending=False)
        )

        # posisi ranking cluster_pred
        rank_position = rank_series.index.tolist().index(cluster_pred)
        if rank_position == 0:
            kategori = "kelompok dengan nilai tertinggi"
        elif rank_position == len(rank_series) - 1:
            kategori = "kelompok dengan nilai terendah"
        else:
            kategori = "kelompok menengah"

        st.markdown(f"""
        - Berdasarkan fitur utama **{main_feat}**, **Cluster {cluster_pred}** termasuk **{kategori}** dibanding cluster lain.
        - Jarak ke centroid **Cluster {cluster_pred}** ≈ **{dists[cluster_pred]:.3f}** di ruang PCA.
        - Semakin kecil jarak ke centroid, semakin mirip karakteristik data baru ini dengan pola negara-negara dalam cluster tersebut.
        - Nilai confidence relatif: **{confidence*100:,.1f}%** (semakin tinggi → semakin yakin model terhadap cluster ini).
        """)

        st.markdown("""
        **Contoh narasi yang bisa kamu tulis di laporan:**

        > Berdasarkan skenario nilai input yang dimasukkan, data baru dipetakan ke dalam **Cluster X**. 
        > Hal ini menunjukkan bahwa pola investasi dan karakteristik tenaga kerja pada skenario tersebut 
        > paling mendekati negara-negara yang tergabung dalam cluster yang sama. 
        > Jika dilihat dari profil rata-rata cluster, Cluster X berada pada kategori 
        > (tinggi/menengah/rendah) dari sisi indikator utama, sehingga skenario ini dapat 
        > diinterpretasikan sebagai (negara/skenario) dengan kontribusi investasi yang 
        > relatif (kuat/sedang/lemah).
        """)

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# TAB 4 – DOWNLOAD & RINGKASAN
# ====================================================
with tab4:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.header("4️⃣ Download & Ringkasan")

    st.subheader("📥 Unduh Hasil Clustering")
    download_csv_button(df_clustered)

    st.subheader("🔍 Ringkasan Singkat untuk Laporan")
    n_cluster = df_clustered["Cluster"].nunique()
    st.markdown(f"""
    - Jumlah negara investor: **{df_clustered.shape[0]}**
    - Jumlah fitur yang digunakan: **{len(selected_features)}**
    - Jumlah cluster (K-Means): **{n_cluster}**
    - Silhouette Score: **{sil:.3f}**
    - Davies-Bouldin Index: **{dbi:.3f}**
    """)

    st.markdown("""
    **Contoh narasi laporan:**

    > Data Penanaman Modal Asing (PMA) Kota Surabaya dikelompokkan menggunakan algoritma K-Means 
    > setelah melalui tahap winsorizing dan normalisasi Z-Score. Reduksi dimensi dilakukan 
    > dengan Principal Component Analysis (PCA) menjadi dua komponen utama yang mampu menjelaskan 
    > sebagian besar keragaman data. Berdasarkan analisis Elbow dan Silhouette, pemilihan jumlah cluster K 
    > dilakukan pada nilai yang memberikan keseimbangan antara kompaknya cluster dan pemisahan antarcluster. 
    > Hasil clustering menunjukkan adanya perbedaan yang jelas antara negara dengan kontribusi investasi tinggi 
    > dan negara dengan kontribusi relatif rendah, yang tercermin dari rata-rata nilai investasi, jumlah proyek, 
    > serta serapan tenaga kerja (TKI dan TKA) pada masing-masing cluster.
    """)

    st.caption("Narasi ini bisa kamu modifikasi sesuai gaya bahasa dan format tugas dosen.")

    st.markdown('</div>', unsafe_allow_html=True)


# ===================== END OF APP =====================
