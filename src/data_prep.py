# src/data_prep.py
import pandas as pd

def clean_and_engineer_features(file_path):
    """
    Carga el dataset de PaySim, filtra tipos relevantes y crea variables de tiempo.
    """
    print("Cargando datos...")
    df = pd.read_csv(file_path)
    
    # ünicamente se consideran los tipos de movimientos TRANSFER y CASH_OUT
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    # Time-Based Features
    print("Creando variables temporales...")
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7
    
    # Business Flags
    print("Creando banderas de negocio...")
    df['flag_empty_origin'] = ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)).astype(int)
    
    # Transformamos el tipo categórico ('TRANSFER'/'CASH_OUT') a binario (1 y 0)
    df['type_is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    
    # Eliminar columnas inútiles para el ML nameOrig y nameDest son IDs (strings), no le sirven al modelo directamente.
    cols_to_drop = ['type', 'nameOrig', 'nameDest', 'isFlaggedFraud', 'step', 'newbalanceOrig', 'newbalanceDest', 'oldbalanceDest']
    df = df.drop(columns=cols_to_drop)
    
    print(f"Dataset final listo. Dimensiones: {df.shape}")
    return df

if __name__ == "__main__":
    df_clean = clean_and_engineer_features('./data/paysim.csv')
    print(df_clean.head())