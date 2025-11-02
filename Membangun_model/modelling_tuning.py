import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import os
import matplotlib.pyplot as plt

# --- Konfigurasi ---
PROCESSED_DATA_PATH = "../preprocessing/telco_churn_preprocessed.csv"
PREPROCESSOR_PATH = "../preprocessing/preprocessor.joblib" # Kita butuh ini untuk artifact
MLRUNS_DIR = "file:./mlruns"
EXPERIMENT_NAME = "Telco Churn Tuning"

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

def hyperparameter_tuning(X_train, y_train):
    """Melakukan hyperparameter tuning untuk Random Forest."""
    print("Memulai Hyperparameter Tuning (GridSearchCV)...")
    
    # Tentukan model
    rf = RandomForestClassifier(random_state=42)
    
    # Tentukan grid parameter (ini bisa Anda perluas)
    # Kita buat grid kecil agar cepat
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Gunakan GridSearchCV
    # cv=3 (3-fold cross-validation)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    print(f"Parameter terbaik ditemukan: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_and_log(model, best_params, X_train, y_train, X_test, y_test):
    """Mengevaluasi model dan mencatat semuanya ke MLflow secara manual."""
    
    print("Mengevaluasi model dan melakukan manual logging...")
    
    # Prediksi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Hitung Metrik (Train)
    metrics_train = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_precision": precision_score(y_train, y_pred_train),
        "train_recall": recall_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train)
    }
    
    # Hitung Metrik (Test)
    metrics_test = {
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test)
    }
    
    # --- MLflow Manual Logging (Kriteria 2 Skilled) ---
    
    # 1. Log Parameter (Parameter terbaik dari tuning)
    mlflow.log_params(best_params)
    
    # 2. Log Metrik (Train & Test)
    mlflow.log_metrics(metrics_train)
    mlflow.log_metrics(metrics_test)
    
    # 3. Log Artifact (Model)
    # 'model' adalah nama folder artifact di dalam MLflow
    mlflow.sklearn.log_model(model, "model")
    
    # 4. Log Artifact (Preprocessor) - Ini PENTING
    # Kita mencatat preprocessor yang digunakan untuk melatih model ini
    mlflow.log_artifact(PREPROCESSOR_PATH, "preprocessor")

    # 5. Log Artifact (Confusion Matrix) - Tambahan
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.title("Test Confusion Matrix")
    
    # Simpan gambar dan log sebagai artifact
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    
    print("Manual logging selesai.")
    return metrics_test

def main():
    """Fungsi utama untuk pipeline tuning."""
    
    # Atur URI dan Eksperimen
    os.environ["MLFLOW_TRACKING_URI"] = MLRUNS_DIR
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    df = load_data(PROCESSED_DATA_PATH)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)

    # Mulai MLflow Run
    with mlflow.start_run(run_name="RandomForest_Tuned_ManualLog") as run:
        print("Run ID:", run.info.run_id)
        
        # 3. Latih & Tuning
        best_model, best_params = hyperparameter_tuning(X_train, y_train)
        
        # 4. Evaluasi & Log
        metrics = evaluate_and_log(best_model, best_params, X_train, y_train, X_test, y_test)
        
        print(f"\nModel (Random Forest Tuned) dilatih.")
        print(f"Metrik Test: {metrics}")
        
if __name__ == "__main__":
    main()