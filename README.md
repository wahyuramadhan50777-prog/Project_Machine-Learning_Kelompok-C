# 📱 Mobile Price Classification — Machine Learning Project

**ANGGOTA KELOMPOK**
- Sherly Steffiyani Askarilia (K1D024042)
- Wahyu Putra Ramadhan (K1D024072)
- Nata Naila Ramadani (K1D024073)
- Rohmatunnisa Agnar Azarine (K1D024074)

## Deskripsi Project

Mobile Price Classification merupakan proyek machine learning yang dirancang untuk memprediksi rentang harga ponsel berdasarkan spesifikasi teknisnya secara otomatis. Sistem ini mencakup proses eksplorasi data, visualisasi, seleksi fitur, pelatihan model, hingga evaluasi performa menggunakan berbagai algoritma klasifikasi.

Project ini dibangun menggunakan Python dan menerapkan konsep supervised learning dengan membandingkan beberapa model klasifikasi untuk menemukan model terbaik.

### Fitur Utama

- Eksplorasi dan analisis dataset ponsel
- Visualisasi distribusi fitur dan korelasi
- Penanganan outlier dengan metode IQR
- Seleksi fitur dengan SelectKBest (Mutual Information)
- Pelatihan model: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning dengan GridSearchCV
- Evaluasi model dengan Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## Cara Menjalankan Script

### 1. Persiapan

Pastikan sudah terinstall:

- Python 3.8+
- Jupyter Notebook / Google Colab

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Siapkan Dataset

Letakkan file `train.csv` di direktori yang sama dengan notebook, atau sesuaikan path:

```python
df = pd.read_csv('/content/train.csv')
```

### 3. Jalankan Notebook

Buka file `projectml1_aseli_fix.ipynb` di Google Colab atau Jupyter Notebook lalu jalankan semua sel secara berurutan.

### 4. Verifikasi Dataset

```python
df.info()
df.describe()
df['price_range'].value_counts()
```

---

## Dataset

### Informasi Umum

| Keterangan | Detail |
|------------|--------|
| Jumlah data | 2.000 baris |
| Jumlah fitur | 20 fitur + 1 target |
| Missing values | Tidak ada |
| Duplikat | Tidak ada |
| Keseimbangan kelas | Sempurna (25% tiap kelas) |

### Target Variable

| Kelas | Kategori |
|-------|----------|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

### Fitur Numerik

| Fitur | Deskripsi |
|-------|-----------|
| `battery_power` | Kapasitas baterai (mAh) |
| `clock_speed` | Kecepatan prosesor (GHz) |
| `fc` | Megapiksel kamera depan |
| `int_memory` | Memori internal (GB) |
| `m_dep` | Ketebalan ponsel (cm) |
| `mobile_wt` | Berat ponsel (gram) |
| `n_cores` | Jumlah core prosesor |
| `pc` | Megapiksel kamera utama |
| `px_height` | Resolusi tinggi piksel |
| `px_width` | Resolusi lebar piksel |
| `ram` | RAM (MB) |
| `sc_h` | Tinggi layar (cm) |
| `sc_w` | Lebar layar (cm) |
| `talk_time` | Waktu bicara (jam) |

### Fitur Kategorikal

| Fitur | Deskripsi |
|-------|-----------|
| `blue` | Bluetooth (0/1) |
| `dual_sim` | Dual SIM (0/1) |
| `four_g` | Dukungan 4G (0/1) |
| `three_g` | Dukungan 3G (0/1) |
| `touch_screen` | Layar sentuh (0/1) |
| `wifi` | WiFi (0/1) |

---

## Alur Pengerjaan

```
1. Import Library
       ↓
2. Load & Eksplorasi Dataset
       ↓
3. Visualisasi (Histogram, Boxplot, Scatter Plot, Heatmap, Bar Chart, Pie Chart)
       ↓
4. Penanganan Outlier (IQR Capping)
       ↓
5. Seleksi Fitur (SelectKBest — Mutual Information, k=10)
       ↓
6. Split Data (Train 80% / Test 20%, Stratified)
       ↓
7. Scaling (StandardScaler — fit hanya pada data train)
       ↓
8. Pelatihan & Evaluasi Model
       ↓
9. Perbandingan Model & Pemilihan Model Terbaik
```

---

## Model yang Digunakan

### 1. Logistic Regression

```python
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
```

### 2. Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5,
                             min_samples_leaf=2, max_features='sqrt', random_state=42)
```

### 3. Random Forest (Top 6 Fitur)

Fitur terpilih: `ram`, `battery_power`, `px_width`, `px_height`, `sc_w`, `m_dep`

### 4. XGBoost (Default)

```python
xgb = XGBClassifier(random_state=42)
```

### 5. XGBoost (GridSearchCV)

```python
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}
```

Best params: `learning_rate=0.2`, `max_depth=3`, `min_child_weight=5`, `n_estimators=200`

---

## Hasil Evaluasi Model

### Perbandingan Performa (Data Testing)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.9675** | **0.9674** | **0.9675** | **0.9673** | **0.9986** |
| Random Forest | 0.9175 | 0.9172 | 0.9175 | 0.9173 | 0.9889 |
| Random Forest (Top 6) | 0.9300 | 0.9303 | 0.9300 | 0.9301 | 0.9930 |
| XGBoost (Default) | 0.9475 | 0.9473 | 0.9475 | 0.9473 | 0.9955 |
| XGBoost (GridSearchCV) | 0.9450 | 0.9450 | 0.9450 | 0.9448 | 0.9955 |

### Deteksi Overfitting (Train vs Test Accuracy)

| Model | Train Acc | Test Acc | Gap |
|-------|-----------|----------|-----|
| Logistic Regression | 0.9544 | 0.9675 | -0.0131  |
| Random Forest | 0.9969 | 0.9175 | 0.0794  |
| Random Forest (Top 6) | 0.9969 | 0.9300 | 0.0669  |
| XGBoost (Default) | 1.0000 | 0.9475 | 0.0525 |
| XGBoost (GridSearchCV) | 0.9994 | 0.9450 | 0.0544 |

---

## Model Terbaik

**✅ Logistic Regression** dipilih sebagai model terbaik karena:

- Akurasi testing **tertinggi (96.75%)** di antara semua model
- **Tidak ada overfitting**  gap train vs test bernilai negatif (-0.0131), artinya model justru lebih baik di data baru
- Nilai Precision, Recall, dan F1-Score sangat konsisten di semua kelas (≈0.967)
- Model paling **stabil dan generalisasi** dibanding model lainnya

---

## Fitur Terpenting

Berdasarkan analisis korelasi, mutual information, dan feature importance Random Forest:

| Peringkat | Fitur | Importance (RF) | Keterangan |
|-----------|-------|-----------------|------------|
| 1 | `ram` | 0.6565 | Prediktor paling dominan |
| 2 | `battery_power` | 0.0982 | Kapasitas baterai |
| 3 | `px_width` | 0.0727 | Resolusi lebar piksel |
| 4 | `px_height` | 0.0672 | Resolusi tinggi piksel |
| 5 | `sc_w` | 0.0300 | Lebar layar fisik |
| 6 | `m_dep` | 0.0273 | Ketebalan ponsel |

> **Insight utama:** RAM memiliki korelasi **0.92** dengan `price_range` — jauh melampaui semua fitur lainnya. Semakin tinggi RAM, semakin tinggi kategori harga ponsel.

---

## Struktur Folder

```text
mobile-price-classification/
│
├── projectml1_aseli_fix.ipynb   # Notebook utama (EDA + Modeling)
├── train.csv                    # Dataset training
└── README.md
```

---

## Teknologi yang Digunakan

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest, SelectKBest, StandardScaler)
- XGBoost
- Google Colab / Jupyter Notebook

---

## Jumlah Model

Project ini mengimplementasikan **5 model klasifikasi**, yaitu:

- Logistic Regression
- Random Forest (All Features)
- Random Forest (Top 6 Features)
- XGBoost Default
- XGBoost dengan GridSearchCV

---

## Author

Project Machine Learning — Klasifikasi Rentang Harga Ponsel  
Dibuat untuk memenuhi tugas praktikum/ujian Machine Learning.
