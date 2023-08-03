# Copyryht @magno000 
# Este Codigo es completamente funcional


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")

# Cargar los datos del conjunto de datos de Kaggle (reemplaza 'hmeq_data.csv' con la ruta correcta del archivo)
df = pd.read_csv('hmeq.csv')

# Preprocesamiento de datos
df = df.dropna()  # Eliminar filas con valores faltantes
df['BAD'] = df['BAD'].astype(int)  # Convertir la variable objetivo en valores enteros

# Variables categóricas
df = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)

# Variables numéricas
numeric_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Añadir columnas 'REASON_DebtCon' y 'REASON_HomeImp' al conjunto de entrenamiento
df['REASON_DebtCon'] = 0
df['REASON_HomeImp'] = 0

# Separar características y etiquetas
features = df.drop('BAD', axis=1)
labels = df['BAD']

# Divide los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

rf_m = RandomForestClassifier(n_estimators=100, random_state=42)
model_lin_reg = LinearRegression()
model_log_reg = LogisticRegression()
svc_m = SVC()

# Entrena el modelo
rf_mo = rf_m.fit(X_train, y_train)
lin_reg = model_lin_reg.fit(X_train, y_train)
log_reg = model_log_reg.fit(X_train, y_train)
svc_mo = svc_m.fit(X_train, y_train)

with open('rf_mo.pkl', 'wb') as modelado_rf:
    pickle.dump(rf_mo, modelado_rf)

with open('lin_reg.pkl', 'wb') as modelado_linear:
    pickle.dump(lin_reg, modelado_linear)

with open('log_reg.pkl', 'wb') as modelado_logistico:
    pickle.dump(log_reg, modelado_logistico)   

with open('svc_mo.pkl', 'wb') as modelado_svc:
    pickle.dump(svc_mo, modelado_svc)    

# Función para calcular los KPIs
def calculate_kpis(dataframe, model):
    # Crear un dataframe con la entrada del usuario
    user_data = dataframe.copy()
    
    # Codificar las variables categóricas
    if user_data['reason'][0] == 'DebtCon':
        user_data['REASON_DebtCon'] = 1
        user_data['REASON_HomeImp'] = 0
    else:
        user_data['REASON_DebtCon'] = 0
        user_data['REASON_HomeImp'] = 1

    if user_data['job'][0] in ['Office', 'Other', 'ProfExe', 'Sales', 'Self']:
        user_data[f'JOB_{user_data["job"][0]}'] = 1
        user_data['JOB_Office'] = 0
        user_data['JOB_Other'] = 0
        user_data['JOB_ProfExe'] = 0
        user_data['JOB_Sales'] = 0
        user_data['JOB_Self'] = 0
    
    # Asegurar que todas las columnas existan en user_data
    for col in features.columns:
        if col not in user_data.columns:
            user_data[col] = 0

    # Reordenar las columnas para que coincidan con el orden de entrenamiento
    user_data = user_data[features.columns]

    # Escalar las variables numéricas
    user_data[numeric_features] = scaler.transform(user_data[numeric_features])    

    # Cargar el modelo entrenado desde el archivo
    if model == 'Random Forest':
        with open('rf_mo.pkl', 'rb') as modelado_rf:
            rf_mo = pickle.load(modelado_rf)
        prediction = rf_mo.predict(user_data)
    elif model == 'Linear Regression':
        with open('lin_reg.pkl', 'rb') as modelado_linear:
            lin_reg = pickle.load(modelado_linear)
        prediction = lin_reg.predict(user_data)
    elif model == 'Logistic Regression':
        with open('log_reg.pkl', 'rb') as modelado_logistico:
            log_reg = pickle.load(modelado_logistico)
        prediction = log_reg.predict(user_data)
    else:
        with open('svc_mo.pkl', 'rb') as modelado_svc:
            svc_mo = pickle.load(modelado_svc)
        prediction = svc_mo.predict(user_data)

    return prediction[0]

# Diseño de la aplicación Streamlit
def main():
    st.title('Scoring de Riesgo para Concesión de Créditos')

    # st.subheader('Introduce los datos del solicitante:')

    st.sidebar.header('Introduce los datos del solicitante:')

    def user_input_parameters():
        loan = st.sidebar.slider('Importe solicitado del préstamo', min_value=0, value=100000)
        reason = st.sidebar.selectbox('Finalidad del préstamo', ['DebtCon', 'HomeImp'])
        job = st.sidebar.selectbox('Trabajo', ['Office', 'Other', 'ProfExe', 'Sales', 'Self'])
        mortdue = st.sidebar.slider('Valor de la hipoteca existente', min_value=0, value=50000)
        value = st.sidebar.slider('Valor de la propiedad', min_value=0, value=100000)
        yoj = st.sidebar.slider('Años en el trabajo actual', min_value=0, value=5)
        derog = st.sidebar.slider('Número de informes desfavorables', min_value=0, value=0)
        delinq = st.sidebar.slider('Número de pagos atrasados', min_value=0, value=0)
        clage = st.sidebar.slider('Antigüedad de la línea de crédito más antigua', min_value=0, value=100)
        ninq = st.sidebar.slider('Número de líneas de crédito recientes', min_value=0, value=0)
        clno = st.sidebar.slider('Número de líneas de crédito', min_value=0, value=5)
        debtinc = st.sidebar.slider('Ratio deuda-ingresos', min_value=0, value=30)

        data = {'loan': loan,
                'reason': reason,
                'job': job,                
                'mortdue': mortdue,
                'value': value,
                'yoj': yoj,
                'derog': derog,
                'delinq': delinq,
                'clage': clage,
                'ninq': ninq,
                'clno': clno,
                'debtinc': debtinc,
                }
        condicionantes = pd.DataFrame(data, index=[0])
        return condicionantes
    
    dataframe = user_input_parameters()
       

    # Escoger el modelo predictivo
    option = ['Random Forest', 'Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox("Qué modelo eliges?", option)

    # Visualiza en una tabla los datos introducidos por el usuario
    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(dataframe)

    # Inicializar la variable prediction con un valor predeterminado
    prediction = None

    if st.sidebar.button('Calcular riesgo'):
        prediction = calculate_kpis(dataframe, model)
        if prediction == 0:
            st.success('El préstamo fue aprobado. No hay riesgo de incumplimiento.')
        else:
            st.success('El préstamo fue rechazado. Existe riesgo de incumplimiento.')

# Arrancar la aplicación Streamlit
if __name__ == '__main__':
    main()
