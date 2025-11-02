import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import argparse

RAW_DATA_PATH = "Telco Customer Churn.csv"
OUTPUT_FOLDER = "preprocessing"
PROCESSED_DATA_PATH = os.path.join(OUTPUT_FOLDER, "telco_churn_preprocessed.csv")
PREPROCESSOR_PATH = os.path.join(OUTPUT_FOLDER, "preprocessor.joblib")

def load_data(path):
    """Memuat data mentah dari file CSV."""
    print(f"Memuat data mentah dari {path}...")
    return pd.read_csv(path)

def clean_data(df):
    """Melakukan pembersihan data awal."""
    print("Memulai pembersihan data...")
    df_clean = df.copy()
    
    # 1. Hapus customerID
    df_clean = df_clean.drop('customerID', axis=1)
    
    # 2. Konversi TotalCharges dan tangani missing values
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean.dropna(inplace=True)
    
    print(f"Data bersih, jumlah baris: {len(df_clean)}")
    return df_clean

def define_preprocessor(df):
    """Mendefinisikan fitur dan pipeline preprocessor."""
    
    # 1. Pisahkan fitur dan target
    df_prep = df.copy()
    df_prep['Churn'] = df_prep['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df_prep.drop('Churn', axis=1)
    y = df_prep['Churn']
    
    # 2. Identifikasi tipe kolom
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    print(f"Fitur Numerik: {numerical_features}")
    print(f"Fitur Kategorikal: {categorical_features}")

    # 3. Buat pipeline transformer
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, X, y

def process_and_save(preprocessor, X, y, output_data_path, output_preprocessor_path):
    """Menjalankan preprocessing dan menyimpan hasilnya."""
    
    print("Menjalankan fit_transform pada data...")
    # 'fit_transform' untuk data pelatihan
    X_processed = preprocessor.fit_transform(X)
    
    # Dapatkan nama fitur baru setelah One-Hot Encoding
    encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(X.columns[X.columns.isin(preprocessor.transformers_[1][2])])
    
    # Gabungkan nama fitur
    numerical_features = X.columns[X.columns.isin(preprocessor.transformers_[0][2])]
    new_feature_names = list(numerical_features) + list(encoded_cat_features)
    
    # Buat DataFrame hasil proses
    df_processed = pd.DataFrame(X_processed, columns=new_feature_names)
    
    # Gabungkan kembali dengan target
    df_final = pd.concat([df_processed, y.reset_index(drop=True)], axis=1)
    
    # Simpan data yang sudah diproses
    df_final.to_csv(output_data_path, index=False)
    print(f"Data yang sudah diproses disimpan di: {output_data_path}")
    
    # Simpan preprocessor yang sudah di-fit
    joblib.dump(preprocessor, output_preprocessor_path)
    print(f"Preprocessor yang sudah di-fit disimpan di: {output_preprocessor_path}")

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline preprocessing."""
    
    # Pastikan folder output ada
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Load
    df_raw = load_data(RAW_DATA_PATH)
    
    # 2. Clean
    df_clean = clean_data(df_raw)
    
    # 3. Define
    preprocessor, X, y = define_preprocessor(df_clean)
    
    # 4. Process & Save
    process_and_save(preprocessor, X, y, PROCESSED_DATA_PATH, PREPROCESSOR_PATH)
    
    print("\nOtomatisasi preprocessing selesai.")

if __name__ == "__main__":
    main()