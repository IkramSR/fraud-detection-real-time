# api/main.py
import os
import pandas as pd
import uvicorn
import mlflow.xgboost
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient

class TransactionIn(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float

class TransactionOut(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float 

MODEL = None
THRESHOLD = 0.5  # Valor por defecto por seguridad

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, THRESHOLD
    print("Iniciando API y conectando a Azure Model Registry...")
    
    # Cargar variables de entorno y autenticación
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("AZURE_MLFLOW_URI"))
    credential = DefaultAzureCredential()
    
    model_name = "FraudDetectionModel_XGBoost"
    client = MlflowClient()
    
    # Buscar qué versión tiene el tag de producción
    production_mv = None
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.tags.get("stage") == "Production":
            production_mv = mv
            break
            
    if not production_mv:
        print("ERROR: No hay ningún modelo con la etiqueta de Producción.")
    else:
        print(f"Descargando Modelo Oficial (Versión {production_mv.version}) desde Azure...")
        try:
            # Descarga el modelo en RAM
            model_uri = f"models:/{model_name}/{production_mv.version}"
            MODEL = mlflow.xgboost.load_model(model_uri)
            
            # Descarga el umbral dinámico de ese entrenamiento exacto
            run_id = production_mv.run_id
            run_details = client.get_run(run_id)
            # Si por alguna cosa no existe el umbral guardado se usa 0.5 por defecto
            THRESHOLD = float(run_details.data.params.get("best_threshold", 0.5))
            
            print(f"Modelo cargado. Umbral dinámico establecido matemáticamente en: {THRESHOLD:.4f}")
        except Exception as e:
            print(f"Error descargando el modelo: {e}")
            
    yield 

    MODEL = None

app = FastAPI(title="Real-Time Fraud Detection API", lifespan=lifespan)

@app.post("/predict", response_model=TransactionOut)
def predict_fraud(transaction: TransactionIn):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
        
    # Feature Engineering 
    hour_of_day = transaction.step % 24
    day_of_week = (transaction.step // 24) % 7
    flag_empty_origin = 1 if (transaction.oldbalanceOrg == 0 and transaction.amount > 0) else 0
    type_is_transfer = 1 if transaction.type.upper() == 'TRANSFER' else 0
    
    features = pd.DataFrame([{
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'flag_empty_origin': flag_empty_origin,
        'type_is_transfer': type_is_transfer
    }])
    
    fraud_prob = float(MODEL.predict_proba(features)[0][1])
    
    # Se usa un umbral dinámico
    is_fraud = fraud_prob >= THRESHOLD
    
    return TransactionOut(
        is_fraud=is_fraud, 
        fraud_probability=fraud_prob,
        threshold_used=THRESHOLD
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)