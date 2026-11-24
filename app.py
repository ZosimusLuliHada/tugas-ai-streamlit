# ==============================================================================
# APLIKASI MACHINE LEARNING END-TO-END DENGAN STREAMLIT & MONGODB
# Mencakup: Regresi, Klasifikasi, Clustering, Retraining MongoDB, & Komentar Publik
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Scikit-learn: Datasets & Preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scikit-learn: Model Regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Scikit-learn: Model Klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn: Model Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# Scikit-learn: Evaluasi Metrik
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & SESSION STATE
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard AI & Machine Learning",
    page_icon="🤖",
    layout="wide"
)

if "trained" not in st.session_state:
    st.session_state.trained = False
if "results" not in st.session_state:
    st.session_state.results = {}

# ------------------------------------------------------------------------------
# 2. KONEKSI & FUNGSI DATABASE MONGODB
# ------------------------------------------------------------------------------
@st.cache_resource
def init_mongodb():
    """Inisialisasi koneksi ke MongoDB Atlas menggunakan st.secrets."""
    try:
        uri = st.secrets["MONGODB_URI"]
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"Gagal terhubung ke MongoDB: {e}")
        return None

client = init_mongodb()
if client is not None:
    db = client["test_db"]
    iris_col = db["iris_db"]
    comment_col = db["komentar_publik"]

@st.cache_data(ttl=5)
def get_iris_from_mongo():
    """Mengambil data tambahan Iris dari MongoDB."""
    if client is None:
        return pd.DataFrame()
    data = list(iris_col.find())
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["_id"] = df["_id"].astype(str)
    return df

@st.cache_data(ttl=5)
def get_comments_from_mongo():
    """Mengambil daftar komentar publik dari MongoDB diurutkan dari yang terbaru."""
    if client is None:
        return []
    return list(comment_col.find().sort("timestamp", -1))

# ------------------------------------------------------------------------------
# 3. FUNGSI PEMUATAN DATASET (SKLEARN & UCI REPO)
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset(task, dataset_name):
    """Memuat dataset dari scikit-learn atau UCI Machine Learning Repository."""
    if task == "Regresi":
        if dataset_name == "Diabetes (Sklearn)":
            data = datasets.load_diabetes()
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="Target"), None
        elif dataset_name == "California Housing (Sklearn)":
            data = datasets.fetch_california_housing()
            df_x = pd.DataFrame(data.data[:1000], columns=data.feature_names)
            df_y = pd.Series(data.target[:1000], name="Target")
            return df_x, df_y, None

    elif task in ["Klasifikasi", "Clustering"]:
        if dataset_name == "Iris (Sklearn)":
            data = datasets.load_iris()
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="Target"), list(data.target_names)
        elif dataset_name == "Wine (Sklearn)":
            data = datasets.load_wine()
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="Target"), list(data.target_names)
        elif dataset_name == "Breast Cancer (Sklearn)":
            data = datasets.load_breast_cancer()
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="Target"), list(data.target_names)
        elif dataset_name == "Iris (UCI Repo - ID 53)":
            from ucimlrepo import fetch_ucirepo
            iris_uci = fetch_ucirepo(id=53)
            X = iris_uci.data.features
            y_raw = iris_uci.data.targets.iloc[:, 0]
            target_names = sorted(y_raw.unique().tolist())
            y = y_raw.map({name: idx for idx, name in enumerate(target_names)})
            return X, y, target_names

    return None, None, None

# ------------------------------------------------------------------------------
# 4. SIDEBAR: PENGATURAN MODEL MACHINE LEARNING
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Model ML")
st.sidebar.write("Atur parameter di bawah, lalu klik **Train Model**.")

task_type = st.sidebar.radio(
    "🎯 Pilih Jenis Tugas:",
    ["Regresi", "Klasifikasi", "Clustering"]
)

if task_type == "Regresi":
    dataset_options = ["Diabetes (Sklearn)", "California Housing (Sklearn)"]
    model_options = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"]
elif task_type == "Klasifikasi":
    dataset_options = ["Iris (Sklearn)", "Wine (Sklearn)", "Breast Cancer (Sklearn)", "Iris (UCI Repo - ID 53)"]
    model_options = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"]
else:
    dataset_options = ["Iris (Sklearn)", "Wine (Sklearn)", "Iris (UCI Repo - ID 53)"]
    model_options = ["K-Means Clustering", "Agglomerative Clustering"]

selected_dataset = st.sidebar.selectbox("📊 Pilih Dataset:", dataset_options)
selected_model = st.sidebar.selectbox("🧠 Pilih Algoritma:", model_options)

if task_type in ["Regresi", "Klasifikasi"]:
    test_size = st.sidebar.slider("📐 Test Size (Proporsi Uji):", 0.10, 0.50, 0.20, 0.05)
    random_state = st.sidebar.number_input("🎲 Random State:", value=42, step=1)
    n_clusters = None
else:
    n_clusters = st.sidebar.slider("🔢 Jumlah Cluster (K):", 2, 8, 3)
    random_state = st.sidebar.number_input("🎲 Random State:", value=42, step=1)
    test_size = None

train_btn = st.sidebar.button("🚀 Train / Proses Model", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# 5. PROSES TRAINING / CLUSTERING SAAT TOMBOL DIKLIK
# ------------------------------------------------------------------------------
if train_btn:
    with st.spinner("Sedang memproses data dan melatih model..."):
        start_time = time.time()
        X, y, target_names = load_dataset(task_type, selected_dataset)

        if task_type == "Regresi":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=random_state),
                "Lasso Regression": Lasso(random_state=random_state),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
                "Random Forest Regressor": RandomForestRegressor(random_state=random_state)
            }
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.results = {
                "task": task_type, "dataset": selected_dataset, "model_name": selected_model,
                "y_test": y_test, "y_pred": y_pred,
                "train_len": len(X_train), "test_len": len(X_test),
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
                "duration": time.time() - start_time
            }

        elif task_type == "Klasifikasi":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
                "Random Forest Classifier": RandomForestClassifier(random_state=random_state),
                "Support Vector Machine (SVM)": SVC(random_state=random_state)
            }
            model = models[selected_model]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            st.session_state.results = {
                "task": task_type, "dataset": selected_dataset, "model_name": selected_model,
                "y_test": y_test, "y_pred": y_pred, "target_names": target_names,
                "train_len": len(X_train), "test_len": len(X_test),
                "acc": accuracy_score(y_test, y_pred),
                "prec": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "rec": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "cm": confusion_matrix(y_test, y_pred),
                "report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0),
                "duration": time.time() - start_time
            }

        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if selected_model == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)

            cluster_labels = model.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, cluster_labels)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            st.session_state.results = {
                "task": task_type, "dataset": selected_dataset, "model_name": selected_model,
                "X_pca": X_pca, "clusters": cluster_labels, "y_true": y,
                "n_clusters": n_clusters, "total_samples": len(X),
                "silhouette": sil_score,
                "duration": time.time() - start_time
            }

        st.session_state.trained = True

# ------------------------------------------------------------------------------
# 6. TAMPILAN UTAMA (HEADER & 4 TABS)
# ------------------------------------------------------------------------------
st.title("🤖 Aplikasi Machine Learning: Regresi, Klasifikasi & Clustering")
st.markdown("Integrasi **Scikit-Learn**, **UCI Repository**, **MongoDB Atlas (CRUD & Retrain)**, dan **Interaksi Publik**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualisasi Hasil",
    "📈 Evaluation Metrics",
    "🗄️ Database & Retrain (Iris)",
    "💬 Komentar Publik"
])

# ==============================================================================
# TAB 1: VISUALISASI HASIL MODEL
# ==============================================================================
with tab1:
    if not st.session_state.trained:
        st.info("👈 Silakan pilih tugas, dataset, dan model di sidebar, lalu klik **Train / Proses Model**.")
    else:
        res = st.session_state.results
        st.success(f"✅ Proses **{res['model_name']}** selesai dalam **{res['duration']:.4f} detik**!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tugas ML", res["task"])
        c2.metric("Dataset", res["dataset"])
        c3.metric("Algoritma", res["model_name"])
        c4.metric("Waktu Komputasi", f"{res['duration']:.4f} s")

        st.divider()

        if res["task"] == "Regresi":
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Actual vs Predicted (Scatter Plot)")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(res["y_test"], res["y_pred"], alpha=0.7, edgecolors="k")
                min_val = min(res["y_test"].min(), res["y_pred"].min())
                max_val = max(res["y_test"].max(), res["y_pred"].max())
                ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
                ax.set_xlabel("Nilai Aktual")
                ax.set_ylabel("Nilai Prediksi")
                ax.legend()
                st.pyplot(fig)

            with col_b:
                st.subheader("Perbandingan 50 Sampel Pertama")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                limit = min(50, len(res["y_test"]))
                ax2.plot(range(limit), res["y_test"].values[:limit], marker="o", label="Actual")
                ax2.plot(range(limit), res["y_pred"][:limit], marker="x", label="Predicted")
                ax2.set_xlabel("Index Sampel")
                ax2.set_ylabel("Nilai")
                ax2.legend()
                st.pyplot(fig2)

        elif res["task"] == "Klasifikasi":
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=res["target_names"], yticklabels=res["target_names"], ax=ax
            )
            ax.set_xlabel("Prediksi Model")
            ax.set_ylabel("Label Aktual")
            st.pyplot(fig)

        elif res["task"] == "Clustering":
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Hasil Clustering ({res['n_clusters']} Cluster) - Proyeksi PCA 2D")
                fig, ax = plt.subplots(figsize=(6, 4))
                scatter = ax.scatter(res["X_pca"][:, 0], res["X_pca"][:, 1], c=res["clusters"], cmap="viridis", s=50, edgecolors="k")
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                plt.colorbar(scatter, ax=ax, label="Cluster ID")
                st.pyplot(fig)

            with col_b:
                st.subheader("Distribusi Jumlah Anggota per Cluster")
                unique, counts = np.unique(res["clusters"], return_counts=True)
                df_dist = pd.DataFrame({"Cluster": [f"Cluster {u}" for u in unique], "Jumlah Data": counts})
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_dist, x="Cluster", y="Jumlah Data", palette="viridis", ax=ax2)
                st.pyplot(fig2)

# ==============================================================================
# TAB 2: EVALUATION METRICS
# ==============================================================================
with tab2:
    if not st.session_state.trained:
        st.info("👈 Silakan jalankan training terlebih dahulu untuk melihat metrik evaluasi.")
    else:
        res = st.session_state.results
        st.subheader(f"📈 Metrik Evaluasi: {res['model_name']} ({res['task']})")

        if res["task"] == "Regresi":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE (Mean Absolute Error)", f"{res['mae']:.4f}")
            m2.metric("MSE (Mean Squared Error)", f"{res['mse']:.4f}")
            m3.metric("RMSE (Root MSE)", f"{res['rmse']:.4f}")
            m4.metric("R² Score", f"{res['r2']:.4f}")

        elif res["task"] == "Klasifikasi":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{res['acc']*100:.2f}%")
            m2.metric("Precision (Weighted)", f"{res['prec']*100:.2f}%")
            m3.metric("Recall (Weighted)", f"{res['rec']*100:.2f}%")
            m4.metric("F1-Score (Weighted)", f"{res['f1']*100:.2f}%")

            st.subheader("Classification Report Detail")
            df_report = pd.DataFrame(res["report"]).transpose()
            st.dataframe(df_report.style.format("{:.4f}"), use_container_width=True)

        elif res["task"] == "Clustering":
            m1, m2, m3 = st.columns(3)
            m1.metric("Silhouette Score", f"{res['silhouette']:.4f}")
            m2.metric("Jumlah Cluster", res["n_clusters"])
            m3.metric("Total Sampel", res["total_samples"])
            st.caption("Catatan: Silhouette Score berkisar antara -1 hingga 1. Nilai mendekati 1 menandakan pemisahan antar-cluster yang sangat baik.")

# ==============================================================================
# TAB 3: DATABASE MONGODB CRUD & RETRAIN MODEL IRIS
# ==============================================================================
with tab3:
    st.header("🗄️ MongoDB CRUD & Retraining Dataset Iris")

    if client is None:
        st.warning("Koneksi MongoDB belum tersedia. Periksa konfigurasi `MONGODB_URI` di Secrets.")
    else:
        df_mongo = get_iris_from_mongo()

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Data Tambahan di MongoDB", len(df_mongo))
        s2.metric("Database", "test_db")
        s3.metric("Collection", "iris_db")

        st.divider()

        col_form, col_table = st.columns([1, 1.5])

        with col_form:
            st.subheader("➕ Tambah Data Iris Baru")
            with st.form("form_tambah_iris", clear_on_submit=True):
                sepal_l = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
                sepal_w = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
                petal_l = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
                petal_w = st.number_input("Petal Width (cm)", min_value=0.0, max_value=15.0, value=0.2, step=0.1)
                species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])
                submit_iris = st.form_submit_button("💾 Simpan ke MongoDB", type="primary")

                if submit_iris:
                    doc = {
                        "sepal_length": float(sepal_l),
                        "sepal_width": float(sepal_w),
                        "petal_length": float(petal_l),
                        "petal_width": float(petal_w),
                        "species": species,
                        "timestamp": datetime.now()
                    }
                    iris_col.insert_one(doc)
                    st.cache_data.clear()
                    st.success("Data berhasil disimpan ke MongoDB!")
                    st.rerun()

        with col_table:
            st.subheader("📋 Data Tersimpan di MongoDB")
            if df_mongo.empty:
                st.info("Belum ada data tambahan di MongoDB.")
            else:
                st.dataframe(df_mongo, use_container_width=True, height=250)

                del_id = st.selectbox("Pilih `_id` dokumen untuk dihapus:", df_mongo["_id"].tolist())
                if st.button("🗑️ Hapus Dokumen Terpilih"):
                    iris_col.delete_one({"_id": ObjectId(del_id)})
                    st.cache_data.clear()
                    st.warning("Dokumen berhasil dihapus!")
                    st.rerun()

        st.divider()

        st.subheader("🔄 Retrain Model Random Forest (Data Sklearn + Data MongoDB)")
        st.write("Menggabungkan 150 data bawaan Iris (scikit-learn) dengan seluruh data baru dari MongoDB untuk melatih ulang model.")

        if st.button("⚡ Retrain Model Sekarang", type="primary"):
            iris_base = datasets.load_iris()
            df_base = pd.DataFrame(iris_base.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
            df_base["target"] = iris_base.target

            if not df_mongo.empty:
                mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
                df_add = df_mongo[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].copy()
                df_add["target"] = df_add["species"].map(mapping)
                df_add = df_add.drop(columns=["species"]).dropna()
                df_combined = pd.concat([df_base, df_add], ignore_index=True)
            else:
                df_combined = df_base

            X_comb = df_combined[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
            y_comb = df_combined["target"].astype(int)

            X_tr, X_te, y_tr, y_te = train_test_split(X_comb, y_comb, test_size=0.2, random_state=42)
            rf_retrain = RandomForestClassifier(random_state=42)
            rf_retrain.fit(X_tr, y_tr)
            y_pr = rf_retrain.predict(X_te)

            acc_new = accuracy_score(y_te, y_pr)
            st.success(f"🎉 Retraining selesai! Total dataset gabungan: **{len(df_combined)} baris** | Akurasi Uji: **{acc_new*100:.2f}%**")

            col_rep, col_imp = st.columns(2)
            with col_rep:
                st.write("**Classification Report (Model Hasil Retrain):**")
                rep_df = pd.DataFrame(classification_report(
                    y_te, y_pr, target_names=["setosa", "versicolor", "virginica"], output_dict=True, zero_division=0
                )).transpose()
                st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

            with col_imp:
                st.write("**Feature Importance:**")
                imp_df = pd.DataFrame({
                    "Fitur": X_comb.columns,
                    "Importance": rf_retrain.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
                sns.barplot(data=imp_df, x="Importance", y="Fitur", palette="magma", ax=ax_imp)
                st.pyplot(fig_imp)

# ==============================================================================
# TAB 4: ISIAN KOMENTAR PUBLIK (TERSIMPAN DI MONGODB)
# ==============================================================================
with tab4:
    st.header("💬 Buku Tamu & Komentar Publik")
    st.write("Silakan tinggalkan masukan, ulasan, atau catatan pengujian aplikasi ini. Komentar akan tersimpan secara permanen di MongoDB.")

    if client is None:
        st.warning("Koneksi MongoDB belum tersedia.")
    else:
        with st.form("form_komentar_publik", clear_on_submit=True):
            nama_pengirim = st.text_input("Nama / NIM Anda:", placeholder="Contoh: Budi Santoso (682022001)")
            isi_komentar = st.text_area("Komentar / Masukan:", placeholder="Tulis komentar Anda di sini...")
            kirim_komentar = st.form_submit_button("📨 Kirim Komentar", type="primary")

            if kirim_komentar:
                if nama_pengirim.strip() == "" or isi_komentar.strip() == "":
                    st.error("Nama dan isi komentar tidak boleh kosong!")
                else:
                    comment_col.insert_one({
                        "nama": nama_pengirim.strip(),
                        "komentar": isi_komentar.strip(),
                        "timestamp": datetime.now()
                    })
                    st.cache_data.clear()
                    st.success("Terima kasih! Komentar Anda telah disimpan.")
                    st.rerun()

        st.divider()
        st.subheader("Daftar Komentar Terbaru")

        daftar_komentar = get_comments_from_mongo()
        if not daftar_komentar:
            st.info("Belum ada komentar. Jadilah yang pertama berkomentar!")
        else:
            for item in daftar_komentar:
                waktu_str = item["timestamp"].strftime("%d %b %Y, %H:%M:%S") if "timestamp" in item else "-"
                with st.chat_message("user"):
                    st.markdown(f"**{item.get('nama', 'Anonim')}** • *{waktu_str}*")
                    st.write(item.get("komentar", ""))