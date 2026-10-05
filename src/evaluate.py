# src/evaluate.py
import os
import mlflow
import json
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

def evaluate_and_register():
    load_dotenv()
    azure_tracking_uri = os.getenv("AZURE_MLFLOW_URI")
    mlflow.set_tracking_uri(azure_tracking_uri)
    
    client = MlflowClient()
    experiment_name = "Fraud_Detection_Enterprise"
    model_name = "FraudDetectionModel_XGBoost"
    
    print("Conectando a Azure ML para evaluar el último experimento...")
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No se encontró el experimento {experiment_name}.")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("No hay modelos entrenados para evaluar.")
        return
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    auprc = latest_run.data.metrics.get("auprc", 0.0)
    
    print(f"Último modelo (Run ID: {run_id}) tiene un AUPRC de: {auprc:.4f}")
    
    # --- LEER LA PERMUTACIÓN DE VARIABLES ---
    try:
        client.download_artifacts(run_id, "permutation_importance.json", ".")
        with open("permutation_importance.json", "r") as f:
            importance = json.load(f)
            
        top_feature_name = list(importance.keys())[0]
        top_feature_pct = list(importance.values())[0]
        print(f"Auditoría XAI: Variable dominante '{top_feature_name}' representa el {top_feature_pct}% del poder predictivo.")
    except Exception as e:
        print("No se encontró el análisis de variables. Rechazando modelo por opacidad.")
        return

    auprc_threshold = 0.70
    max_feature_weight_allowed = 60.0 # Ninguna variable debe pesar más del 60%
    
    if auprc < auprc_threshold:
        print(f"Rechazado: AUPRC ({auprc:.4f}) es menor al mínimo ({auprc_threshold}).")
        return
        
    if top_feature_pct > max_feature_weight_allowed:
        print(f"RECHAZADO: Data Leakage detectado. La variable '{top_feature_name}' " 
              f"monopoliza el {top_feature_pct}% del modelo. Máximo: {max_feature_weight_allowed}%.")
        return
        
    print(f"Modelo Aprobado: Cumple métricas de rendimiento y explicabilidad.")
    
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client.set_model_version_tag(
        name=model_name,
        version=model_details.version,
        key="stage",
        value="Production"
    )

    print(f"Modelo versión {model_details.version} etiquetado con 'stage: Production' exitosamente en Azure.")

if __name__ == "__main__":
    evaluate_and_register()