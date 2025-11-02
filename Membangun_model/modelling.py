import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# --- Konfigurasi ---
# Path ke data yang sudah diproses dari Kriteria 1
PROCESSED_DATA_PATH = "../preprocessing/telco_churn_preprocessed.csv" 
# (Gunakan '../' untuk naik satu level folder)

# Atur MLflow Tracking URI (simpan di folder lokal 'mlruns')
# MLflow akan otomatis membuat folder 'mlruns' di folder 'Membangun_model'
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
EXPERIMENT_NAME = "Telco Churn Baseline"

def load_data(path):
    """Memuat data yang sudah diproses."""
    print(f"Memuat data dari {path}...")
    return pd.read_csv(path)

def split_data(df, target_col='Churn'):
    """Membagi data menjadi set pelatihan dan pengujian."""
    print("Membagi data...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    """Melatih model Logistic Regression."""
    print("Melatih model Logistic Regression (baseline)...")
    # Kita gunakan solver 'liblinear' yang baik untuk dataset ini
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    """Fungsi utama untuk pipeline pelatihan baseline."""
    
    # Set nama eksperimen
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    df = load_data(PROCESSED_DATA_PATH)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Mulai MLflow Run
    with mlflow.start_run(run_name="LogisticRegression_Autolog_Baseline"):
        
        # 3. Aktifkan MLflow autolog (Kriteria 2 Basic)
        print("Mengaktifkan MLflow autolog untuk scikit-learn...")
        mlflow.sklearn.autolog()
        
        # 4. Latih Model
        model = train_model(X_train, y_train)
        
        # 5. Evaluasi Model (Autolog akan mencatat ini secara otomatis)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel baseline (Logistic Regression) dilatih.")
        print(f"Akurasi Test: {accuracy:.4f}")
        print("\nAutolog MLflow telah mencatat parameter, metrik, dan model.")
        print("Run ID:", mlflow.active_run().info.run_id)

if __name__ == "__main__":
    main()