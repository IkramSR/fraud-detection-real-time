# src/train.py
import os 
from dotenv import load_dotenv
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from data_prep import clean_and_engineer_features
import subprocess
import json
from sklearn.inspection import permutation_importance
from mlflow.models.signature import infer_signature

load_dotenv()

def train_model():
    data_path = './data/paysim.csv'
    df = clean_and_engineer_features(data_path)
    
    # Separar Features (X) y Target (y)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # División Cronológica (80% Train, 20% Test) ya quw el dataset está ordenado temporalmente se hace división directa
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Entrenando con {len(X_train)} filas, validando con {len(X_test)} filas.")
    
    # Cálculo del peso de las clases para el desbalance
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_weight = negatives / positives
    soft_weight = np.sqrt(scale_weight) 
    print(f"Scale Pos Weight original: {scale_weight:.2f} | Suavizado: {soft_weight:.2f}")

    
    # 5. Configurar MLflow (Aquí empieza la magia MLOps)
    azure_tracking_uri = os.getenv("AZURE_MLFLOW_URI")
    if not azure_tracking_uri:
        raise ValueError("Falta variable de entorno AZURE_MLFLOW_URI")
        
    mlflow.set_tracking_uri(azure_tracking_uri)
    mlflow.set_experiment("Fraud_Detection_Enterprise")
    
    with mlflow.start_run(run_name="Azure_Cloud_Training"):
        # Definir hiperparámetros
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr", # Área bajo la curva Precision-Recall
            "scale_pos_weight": soft_weight,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42,
            "tree_method": "hist",
            "lambda": 10
            #"colsample_bytree": 0.8
        }

        try:
            # Intenta ejecutar el comando de NVIDIA para ver si hay GPU
            subprocess.check_output('nvidia-smi')
            params["device"] = "cuda"
            print("GPU NVIDIA detectada. Entrenamiento con CUDA")
        except Exception:
            params["device"] = "cpu"
            print("Entrenando con CPU (No se detectó GPU o CUDA)")
        
        # Registrar hiperparámetros en MLflow
        mlflow.log_params(params)
        
        # Entrenar modelo
        print("Entrenando modelo...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilidad de ser fraude
        p, r, t = precision_recall_curve(y_test, y_pred_proba)
        
        # Evitar división por cero
        f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
        best_t = t[f1s.argmax()]
        print(f"\nMejor umbral por F1: {best_t:.4f} (F1 Máximo: {f1s.max():.4f})")
        
        # Se aplica el umbral a las probabilidades para obtener la clase dura (0 o 1) usando best_t
        y_pred = (y_pred_proba >= best_t).astype(int) 


        y_pred = model.predict(X_test) 
        
        # Calcular Métricas
        auprc = average_precision_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"RESULTADOS:")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Registrar métricas en MLflow
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_param("best_threshold", best_t)
        
        #Permutation importance
        print("Calculando Permutation Importance en el Test Set (esto tomará unos segundos)...")
        # Usamos una muestra para eficiencia, como sugeriste
        idx = X_test.sample(min(100_000, len(X_test)), random_state=42).index
        r = permutation_importance(
            model, X_test.loc[idx], y_test.loc[idx],
            scoring="average_precision", n_repeats=3, random_state=42, n_jobs=-1
        )
        
        # Convertir a porcentajes sobre el total de caída
        total_drop = sum(r.importances_mean)
        perm_pct = {k: round((v / total_drop) * 100, 2) for k, v in zip(X_test.columns, r.importances_mean)}
        # Ordenar de mayor a menor
        perm_pct = dict(sorted(perm_pct.items(), key=lambda item: item[1], reverse=True))
        
        print("\nTop 3 Permutation Importance (Impacto en AUPRC):")
        for i, (k, v) in enumerate(list(perm_pct.items())[:3]):
            print(f"{i+1}. {k}: {v}%")
            
        # Guardar como artefacto JSON en Azure
        with open("permutation_importance.json", "w") as f:
            json.dump(perm_pct, f)
        mlflow.log_artifact("permutation_importance.json")
        # --------------------------------------------------------
        signature = infer_signature(X_test, y_pred)
        # Registrar el modelo real en Azure
        mlflow.xgboost.log_model(model, "model", signature=signature)
        print("\n¡Entrenamiento finalizado y registrado remotamente en Azure ML!")

if __name__ == "__main__":
    train_model()