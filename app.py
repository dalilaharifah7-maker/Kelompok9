import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==============================================================
# KONFIGURASI HALAMAN
# ==============================================================
st.set_page_config(
    page_title="Perbandingan Decision Tree vs Random Forest",
    layout="wide",
)

st.title("Perbandingan Decision Tree dan Random Forest")
st.caption(
    "Upload dataset apa saja (CSV / Excel), pilih kolom target berupa angka, "
    "lalu bandingkan kedua algoritma regresi ini."
)


# ==============================================================
# FUNGSI BANTU
# ==============================================================
def convert_value(value):
    """Ubah teks seperti '1.2m', '350k', '2.5%', '$1,200' menjadi angka.
    Jika gagal diubah, hasilnya NaN."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower().replace("$", "").replace(" ", "")

    if value in ("", "nan", "none", "null", "-"):
        return np.nan

    try:
        if value.endswith("%"):
            return float(value[:-1].replace(",", ""))

        multiplier = 1
        if value.endswith("k"):
            multiplier = 1_000
            value = value[:-1]
        elif value.endswith("m"):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.endswith("b"):
            multiplier = 1_000_000_000
            value = value[:-1]

        return float(value.replace(",", "")) * multiplier
    except ValueError:
        return np.nan


def to_numeric_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.apply(convert_value)


def is_numeric_like(s):
    """True jika kolom (hampir seluruhnya) bisa dibaca sebagai angka."""
    if pd.api.types.is_numeric_dtype(s):
        return True
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    parsed = non_null.apply(convert_value)
    return parsed.notna().mean() >= 0.8


@st.cache_data
def load_data(file_bytes, file_name):
    buf = io.BytesIO(file_bytes)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    # sep=None -> pandas menebak pemisah otomatis (koma, titik koma, tab)
    return pd.read_csv(buf, sep=None, engine="python")


def make_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            list(numeric_features),
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            list(categorical_features),
        ))
    return ColumnTransformer(transformers=transformers)


@st.cache_resource
def train_models(X_train, y_train, numeric_features, categorical_features,
                 max_depth, min_split, min_leaf, n_trees):
    dt = Pipeline(steps=[
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("model", DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
        )),
    ])
    rf = Pipeline(steps=[
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("model", RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return dt, rf


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(y_true, y_pred),
    }


def prediction_table(y_true, y_pred):
    hasil = pd.DataFrame({
        "Nilai Aktual": y_true.values,
        "Nilai Prediksi": y_pred,
    })
    hasil["Selisih"] = (hasil["Nilai Aktual"] - hasil["Nilai Prediksi"]).abs()
    hasil["Persentase Prediksi"] = np.where(
        hasil["Nilai Aktual"] != 0,
        hasil["Nilai Prediksi"] / hasil["Nilai Aktual"] * 100,
        np.nan,
    )
    return hasil.round(2)


def show_metrics(res):
    cols = st.columns(4)
    for col, (name, val) in zip(cols, res.items()):
        col.metric(name, round(val, 4))


def interpret_r2(r2):
    if r2 >= 0.7:
        return "Model memiliki kemampuan prediksi yang cukup baik."
    elif r2 >= 0:
        return "Model memiliki kemampuan prediksi yang masih terbatas."
    return "Model belum memiliki performa prediksi yang baik karena R2 negatif."


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ==============================================================
# SIDEBAR: UPLOAD DAN PARAMETER
# ==============================================================
st.sidebar.header("1. Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload file CSV / Excel", type=["csv", "xlsx", "xls"]
)

st.sidebar.header("2. Parameter Model")
test_size = st.sidebar.slider("Ukuran data testing", 0.1, 0.4, 0.2, 0.05)
max_depth = st.sidebar.slider("max_depth", 2, 15, 5)
min_split = st.sidebar.slider("min_samples_split", 2, 20, 4)
min_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 2)
n_trees = st.sidebar.slider("n_estimators (Random Forest)", 10, 300, 100, 10)
seed = st.sidebar.number_input("random_state pembagian data", value=42, step=1)

if uploaded is None:
    st.info("Silakan upload dataset di sidebar untuk memulai.")
    st.markdown(
        """
**Syarat dataset:**
- Format CSV atau Excel, baris pertama berisi nama kolom.
- Ada minimal satu kolom **angka** yang ingin diprediksi (target).
- Kolom angka boleh memakai format seperti `1.2m`, `350k`, `2.5%` atau `1,200`;
  aplikasi akan mengubahnya otomatis.
- Kolom teks (misalnya negara, kategori) otomatis dianggap kategori.
- Disarankan minimal sekitar 30 baris data.
        """
    )
    st.stop()


# ==============================================================
# LOAD DATA
# ==============================================================
try:
    df = load_data(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

if df.shape[0] < 10 or df.shape[1] < 2:
    st.error("Dataset terlalu kecil. Butuh minimal 10 baris dan 2 kolom.")
    st.stop()

st.success(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
with st.expander("Lihat 5 baris pertama dataset"):
    st.dataframe(df.head())


# ==============================================================
# PILIH TARGET DAN FITUR
# ==============================================================
st.header("Pengaturan Kolom")

numeric_candidates = [c for c in df.columns if is_numeric_like(df[c])]

if not numeric_candidates:
    st.error("Tidak ada kolom berisi angka yang bisa dijadikan target.")
    st.stop()

target = st.selectbox(
    "Kolom target (yang ingin diprediksi, harus berupa angka)",
    numeric_candidates,
    index=len(numeric_candidates) - 1,
)

feature_options = [c for c in df.columns if c != target]

# Kolom teks dengan terlalu banyak nilai unik (misal nama/ID) tidak
# dipilih otomatis karena tidak berguna untuk model.
default_features = []
for c in feature_options:
    if is_numeric_like(df[c]):
        default_features.append(c)
    elif df[c].nunique(dropna=True) <= 50:
        default_features.append(c)

selected_features = st.multiselect(
    "Kolom fitur (yang dipakai untuk memprediksi)",
    feature_options,
    default=default_features,
    help="Kolom teks dengan banyak nilai unik (misalnya nama) sebaiknya tidak dipilih.",
)

if not selected_features:
    st.warning("Pilih minimal satu kolom fitur.")
    st.stop()

numeric_like_selected = [c for c in selected_features if is_numeric_like(df[c])]
force_categorical = st.multiselect(
    "Kolom angka yang ingin diperlakukan sebagai kategori (opsional)",
    numeric_like_selected,
    help="Contoh: kode wilayah atau tahun yang bukan nilai ukuran.",
)

numeric_features = tuple(c for c in numeric_like_selected if c not in force_categorical)
categorical_features = tuple(c for c in selected_features if c not in numeric_features)

high_card = [c for c in categorical_features if df[c].nunique(dropna=True) > 50]
if high_card:
    st.warning(
        "Kolom kategori berikut punya banyak nilai unik dan bisa membuat model "
        f"lambat/kurang akurat: {', '.join(high_card)}"
    )

st.caption(
    f"Fitur numerik: {', '.join(numeric_features) or '-'}  |  "
    f"Fitur kategori: {', '.join(categorical_features) or '-'}"
)


# ==============================================================
# SIAPKAN X DAN y
# ==============================================================
X = df[list(selected_features)].copy()

for c in numeric_features:
    X[c] = to_numeric_series(X[c])

for c in categorical_features:
    X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)

# Buang kolom numerik yang seluruhnya kosong
all_nan = [c for c in numeric_features if X[c].isna().all()]
if all_nan:
    st.warning(f"Kolom berikut kosong semua dan diabaikan: {', '.join(all_nan)}")
    numeric_features = tuple(c for c in numeric_features if c not in all_nan)
    X = X.drop(columns=all_nan)

if not numeric_features and not categorical_features:
    st.error("Tidak ada fitur yang bisa dipakai.")
    st.stop()

y = to_numeric_series(df[target])
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

if len(X) < 20:
    st.error(f"Data valid hanya {len(X)} baris. Butuh minimal 20 baris.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed)
)

c1, c2 = st.columns(2)
c1.metric("Data Training", len(X_train))
c2.metric("Data Testing", len(X_test))

with st.spinner("Melatih model..."):
    dt, rf = train_models(
        X_train, y_train, numeric_features, categorical_features,
        max_depth, min_split, min_leaf, n_trees,
    )

pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)
res_dt = evaluate(y_test, pred_dt)
res_rf = evaluate(y_test, pred_rf)


# ==============================================================
# TAB HASIL
# ==============================================================
tab_dt, tab_rf, tab_cmp, tab_exp, tab_pred = st.tabs([
    "A. Decision Tree",
    "B. Random Forest",
    "C. Perbandingan",
    "D. Uji 5 Percobaan",
    "E. Prediksi Manual",
])

# ---------------- A. DECISION TREE ----------------
with tab_dt:
    st.subheader("Hasil Evaluasi Decision Tree")
    show_metrics(res_dt)

    st.subheader("Tabel Hasil Prediksi")
    tabel_dt = prediction_table(y_test, pred_dt)
    st.dataframe(tabel_dt, use_container_width=True)
    st.download_button(
        "Unduh tabel (CSV)", csv_bytes(tabel_dt), "prediksi_decision_tree.csv", "text/csv"
    )

    st.subheader("Visualisasi Pohon (kedalaman tampil maks. 3)")
    fig, ax = plt.subplots(figsize=(25, 12))
    plot_tree(
        dt.named_steps["model"],
        feature_names=dt.named_steps["preprocessor"].get_feature_names_out(),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(f"Decision Tree untuk Prediksi {target}", fontsize=16)
    st.pyplot(fig)
    plt.close(fig)

# ---------------- B. RANDOM FOREST ----------------
with tab_rf:
    st.subheader("Hasil Evaluasi Random Forest")
    show_metrics(res_rf)

    st.subheader("Tabel Hasil Prediksi")
    tabel_rf = prediction_table(y_test, pred_rf)
    st.dataframe(tabel_rf, use_container_width=True)
    st.download_button(
        "Unduh tabel (CSV)", csv_bytes(tabel_rf), "prediksi_random_forest.csv", "text/csv"
    )

    rf_model = rf.named_steps["model"]
    rf_features = rf.named_steps["preprocessor"].get_feature_names_out()

    st.subheader("Salah Satu Pohon dalam Random Forest")
    fig, ax = plt.subplots(figsize=(25, 12))
    plot_tree(
        rf_model.estimators_[0],
        feature_names=rf_features,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Salah Satu Pohon dalam Random Forest", fontsize=16)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Importance (Top 10)")
    importance = pd.DataFrame({
        "Fitur": rf_features,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    st.dataframe(importance.round(4), use_container_width=True)

    top = importance.sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Fitur"], top["Importance"])
    ax.set_title("Top 10 Feature Importance Random Forest")
    ax.set_xlabel("Nilai Importance")
    ax.set_ylabel("Fitur")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------- C. PERBANDINGAN ----------------
with tab_cmp:
    st.subheader("Tabel Perbandingan Kedua Algoritma")
    perbandingan = pd.DataFrame(
        [res_dt, res_rf], index=["Decision Tree", "Random Forest"]
    )
    st.dataframe(perbandingan.round(4), use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "MSE", "RMSE"]):
        ax.bar(perbandingan.index, perbandingan[metric])
        ax.set_title("Perbandingan " + metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Perbandingan Error Decision Tree dan Random Forest")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(perbandingan.index, perbandingan["R2 Score"])
    ax.set_title("Perbandingan R2 Score")
    ax.set_xlabel("Algoritma")
    ax.set_ylabel("R2 Score")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.values, label="Nilai Aktual", marker="o")
    ax.plot(pred_dt, label="Decision Tree", marker="x")
    ax.plot(pred_rf, label="Random Forest", marker="s")
    ax.set_title("Perbandingan Nilai Aktual dan Hasil Prediksi")
    ax.set_xlabel("Data Testing")
    ax.set_ylabel(target)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Kesimpulan Hasil Perbandingan")
    if res_dt["RMSE"] < res_rf["RMSE"]:
        best_name, best_r2 = "Decision Tree", res_dt["R2 Score"]
    elif res_rf["RMSE"] < res_dt["RMSE"]:
        best_name, best_r2 = "Random Forest", res_rf["R2 Score"]
    else:
        best_name = "Keduanya (RMSE sama)"
        best_r2 = max(res_dt["R2 Score"], res_rf["R2 Score"])

    st.success(f"Algoritma terbaik berdasarkan RMSE: **{best_name}**")

    if res_dt["RMSE"] > 0:
        improvement = (res_dt["RMSE"] - res_rf["RMSE"]) / res_dt["RMSE"] * 100
        if improvement > 0:
            st.write(
                f"Random Forest memiliki RMSE {improvement:.2f}% "
                "lebih rendah dibandingkan Decision Tree."
            )
        elif improvement < 0:
            st.write(
                f"Decision Tree memiliki RMSE {abs(improvement):.2f}% "
                "lebih rendah dibandingkan Random Forest."
            )
        else:
            st.write("Kedua algoritma memiliki RMSE yang sama.")

    st.info(interpret_r2(best_r2))

# ---------------- D. UJI 5 PERCOBAAN ----------------
with tab_exp:
    st.subheader("Pengujian Konsistensi 5 Kali Percobaan")
    st.write(
        "Setiap percobaan memakai random_state pembagian data yang berbeda "
        "(10, 20, 30, 40, 50) dengan parameter model dari sidebar."
    )

    if st.button("Jalankan 5 percobaan"):
        rows = []
        bar = st.progress(0.0)

        for nomor, s in enumerate([10, 20, 30, 40, 50], start=1):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=s
            )
            d, r = train_models(
                Xtr, ytr, numeric_features, categorical_features,
                max_depth, min_split, min_leaf, n_trees,
            )
            e_d = evaluate(yte, d.predict(Xte))
            e_r = evaluate(yte, r.predict(Xte))

            rows.append({
                "Percobaan": nomor,
                "Random State": s,
                "MAE Decision Tree": e_d["MAE"],
                "RMSE Decision Tree": e_d["RMSE"],
                "R2 Decision Tree": e_d["R2 Score"],
                "MAE Random Forest": e_r["MAE"],
                "RMSE Random Forest": e_r["RMSE"],
                "R2 Random Forest": e_r["R2 Score"],
            })
            bar.progress(nomor / 5)

        hasil_df = pd.DataFrame(rows)

        st.subheader("Tabel Hasil 5 Kali Percobaan")
        st.dataframe(hasil_df.round(4), use_container_width=True)

        st.subheader("Rata-rata Hasil Percobaan")
        rata_rata = hasil_df.drop(columns=["Percobaan", "Random State"]).mean()
        st.dataframe(rata_rata.to_frame("Rata-rata").round(4))

        rf_menang = int((hasil_df["RMSE Random Forest"] < hasil_df["RMSE Decision Tree"]).sum())
        dt_menang = int((hasil_df["RMSE Decision Tree"] < hasil_df["RMSE Random Forest"]).sum())
        seri = 5 - rf_menang - dt_menang

        st.subheader("Rekapitulasi Hasil")
        k1, k2, k3 = st.columns(3)
        k1.metric("Random Forest unggul", f"{rf_menang} kali")
        k2.metric("Decision Tree unggul", f"{dt_menang} kali")
        k3.metric("Seri", f"{seri} kali")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hasil_df["Percobaan"], hasil_df["RMSE Decision Tree"],
                marker="o", label="Decision Tree")
        ax.plot(hasil_df["Percobaan"], hasil_df["RMSE Random Forest"],
                marker="s", label="Random Forest")
        ax.set_title("Perbandingan RMSE dalam 5 Kali Percobaan")
        ax.set_xlabel("Percobaan")
        ax.set_ylabel("RMSE")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        dt_avg = hasil_df["RMSE Decision Tree"].mean()
        rf_avg = hasil_df["RMSE Random Forest"].mean()

        st.subheader("Kesimpulan Pengujian Konsistensi")
        st.write(f"Rata-rata RMSE Decision Tree: **{dt_avg:.4f}**")
        st.write(f"Rata-rata RMSE Random Forest: **{rf_avg:.4f}**")

        if rf_avg < dt_avg and rf_menang > dt_menang:
            st.success(
                "Random Forest menunjukkan performa lebih konsisten. "
                "Rata-rata RMSE lebih rendah dan lebih sering unggul."
            )
        elif rf_avg < dt_avg:
            st.warning(
                "Random Forest memiliki rata-rata RMSE lebih rendah. "
                "Namun, keunggulannya belum konsisten pada sebagian besar percobaan."
            )
        elif dt_avg < rf_avg:
            st.warning(
                "Decision Tree memiliki rata-rata RMSE lebih rendah. "
                "Random Forest belum menunjukkan keunggulan berdasarkan RMSE."
            )
        else:
            st.info("Kedua algoritma memiliki rata-rata RMSE yang sama.")

        st.caption(
            "RMSE lebih rendah menunjukkan kesalahan prediksi lebih kecil. "
            "R2 negatif menunjukkan kemampuan prediksi masih belum optimal. "
            "Hasil ini hanya menggambarkan lima pembagian data yang diuji."
        )

# ---------------- E. PREDIKSI MANUAL ----------------
with tab_pred:
    st.subheader(f"Prediksi {target} dari Data Baru")
    st.write("Isi nilai tiap fitur, lalu klik tombol untuk melihat prediksi kedua model.")

    with st.form("form_prediksi"):
        input_values = {}
        cols = st.columns(3)
        for i, c in enumerate(X.columns):
            with cols[i % 3]:
                if c in numeric_features:
                    med = X_train[c].median()
                    default = float(med) if pd.notna(med) else 0.0
                    input_values[c] = st.number_input(c, value=default)
                else:
                    options = sorted(X_train[c].dropna().unique().tolist())
                    input_values[c] = st.selectbox(c, options) if options else np.nan
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        baru = pd.DataFrame([input_values])[list(X.columns)]
        p_dt = float(dt.predict(baru)[0])
        p_rf = float(rf.predict(baru)[0])

        m1, m2 = st.columns(2)
        m1.metric("Prediksi Decision Tree", round(p_dt, 4))
        m2.metric("Prediksi Random Forest", round(p_rf, 4))
