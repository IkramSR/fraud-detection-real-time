# Real-Time Fraud Detection System 

## Contexto de negocio

En la industria financiera, la maduración tardía de las etiquetas (contracargos) y los datasets altamente desbalanceados son los principales retos de la detección de fraude. Este proyecto es una arquitectura MLOps de extremo a extremo que detecta transacciones fraudulentas en tiempo real.

En lugar de un despliegue monolítico estático, la arquitectura **desacopla el entrenamiento del modelo del servicio de la API**: Azure Machine Learning actúa como Model Registry remoto, y Docker/FastAPI se encarga del despliegue.

## Arquitectura y pipeline

- **Ingeniería de variables:** variables temporales y de comportamiento extraídas del dataset PaySim para capturar patrones como la toma de control de cuentas y el vaciado de saldo. El análisis exploratorio está en [`notebooks/01_eda_paysim.ipynb`](notebooks/01_eda_paysim.ipynb).
- **Entrenamiento (`src/train.py`):** aprendizaje sensible al costo con XGBoost, manejado mediante `scale_pos_weight`. Los hiperparámetros y las métricas se registran de forma remota en Azure ML (MLflow).
- **Control de calidad (`src/evaluate.py`):** script automatizado emulando un sistema CI/CD, que evalúa el modelo candidato. Usa *Permutation Importance* (XAI) para rechazar modelos con posible fuga de datos (si una sola variable concentra más del 60 % de la importancia) y solo promueve a producción los modelos que cumplen el AUPRC mínimo, dotando de explicabilidad al sistema.
- **Servicio (`api/main.py`):** contenedor Docker ligero con FastAPI, desplegado en Azure Container Instances (ACI). Al iniciar, descarga dinámicamente el último modelo en producción y su umbral de decisión optimizado (`best_t`) desde el registro de Azure.

## Decisiones clave de ingeniería

1. **Ajuste del umbral de decisión.** El umbral por defecto (0.5) es arbitrario. El pipeline calcula el umbral óptimo con la curva Precision-Recall para maximizar el F1-Score, y lo registra como parámetro del modelo para que la API lo consuma.
2. **Tratamiento de la fuga de datos.** Se eliminaron las variables posteriores a la transacción (`newbalanceOrig`, `newbalanceDest`, `errorBalanceOrig`), que en PaySim funcionan como una firma sintética del fraude y que un endpoint real no conocería al momento de decidir. Se auditó además `flag_empty_account`: está disponible antes de la transacción, pero replica la regla del simulador (el atacante siempre vacía la cuenta). La comparación de versiones se resume abajo.
3. **Despliegue desacoplado.** La imagen de Docker pasó de ~2.4 GB a ~700 MB al no incluir el modelo en la imagen: este se carga en memoria al arrancar mediante `DefaultAzureCredential`.

## Resultados

| Versión | Variables | AUPRC | Variable dominante (permutation importance) |
|---|---|---|---|
| v1 | Incluye variables post-transacción y `flag_empty_account` | 0.9999 | `flag_empty_account` (98.76 %) |
| v2 | Sin variables post-transacción ni `flag_empty_account` | 0.7821 | `oldbalanceOrg` (45.84 %) |

La caída de 0.9999 a 0.7821 es el efecto esperado de quitar la fuga de datos y la regla trivial del simulador. La v2 (código en este repositorio) es la cifra más honesta, aunque con el umbral por defecto mostró recall alto y precision baja, razón por la cual el pipeline ajusta el umbral en esta última versión.

> **Limitación.** PaySim es un dataset sintético: los patrones de hora, saldo y comportamiento del atacante son producto del simulador pero funciona como proyecto ilustrativo de combinación de infraestructura en la nube y explicabilidad matemática.

## 📸 Vistas del sistema

**Model Registry en Azure ML**

Modelo etiquetado como "Producción" al cumplir las reglas de aceptabilidad

<img width="1698" height="493" alt="image" src="https://github.com/user-attachments/assets/ad9719b6-e261-498c-96d5-9426b6e4b90b" />



**Control de calidad con Permutation Importance y Metrics**

Antes y después de establecer condiciones para evitar sobreajuste y fuga de datos

<img width="1189" height="515" alt="Captura de pantalla 2026-10-02 214851" src="https://github.com/user-attachments/assets/646a267d-8739-4390-92f4-ad68f4784eb4" />


<img width="1262" height="635" alt="image" src="https://github.com/user-attachments/assets/db42488a-d61b-4cd3-85fb-665145e93436" />



**Azure Container y Predicción en tiempo real vía FastAPI**

<img width="1509" height="788" alt="Captura de pantalla 2026-10-03 212359" src="https://github.com/user-attachments/assets/2260bb7a-2a24-49f0-8ced-33152fca7377" />

<img width="1903" height="818" alt="Captura de pantalla 2026-10-03 212549" src="https://github.com/user-attachments/assets/60321f12-b8ca-482d-93df-2a44d3ce5be0" />


## Cómo ejecutarlo localmente

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/tu-usuario/fraud_detection_project.git
cd fraud_detection_project
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```text
AZURE_MLFLOW_URI=azureml://<uri-de-tu-workspace-en-azure>
```

> **Nota:** requiere una sesión activa de Azure CLI (`az login`) con acceso de lectura al workspace.

### 3. Entrenar y evaluar (simulación de CI/CD)

```bash
python src/train.py
python src/evaluate.py
```

### 4. Levantar la API

```bash
docker build -t fraud-api-v2 .
docker run -p 8000:8000 --env-file .env fraud-api-v2
```

Abre [http://localhost:8000/docs](http://localhost:8000/docs) para probar el endpoint.
