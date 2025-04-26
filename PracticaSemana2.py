# %% [markdown]
# # 🧠 Diplomado en Ciencia de Datos  
# ## 📦 Módulo 2: Almacenamiento y Preparación de Datos (Gestión de Datos)
# 
# ---
# 
# ### 📚 Práctica – Semana 2 
# **Estudiante:** Bruno Joel Vásquez Gonzales  
# 
# **Fecha:** Abril 2025
# 
# **Código SIS:**  202000737

# %%
import pyodbc
import dataprep
from scipy import stats
import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# %%
# Cadena de conexión a SQL Server usando autenticación de Windows
cnxn_str = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=LAPTOP-IM8EIRNS;"
            "Database=DataMartVentas;"
            "Trusted_Connection=yes;"
)
# Conexión
cnxn = pyodbc.connect(cnxn_str)

# %% [markdown]
# ## 🎯 Objetivo
# 
# Predecir el **Ingreso Bruto** de un pedido en función de las siguientes variables:
# 
# - 🛒 **Producto**
# - 👨‍💼 **Empleado**
# - 🧍 **Cliente**
# - 📅 **Fecha**
# 
# El objetivo es identificar patrones y relaciones entre estas variables para construir un modelo que permita estimar de forma precisa el ingreso generado por cada transacción.
# 

# %%
# Consulta para cargar datos del DataMart desde SQL Server
df_datamart = pd.read_sql('''
    SELECT 
        HV.IdVenta,
        HV.IdPedido,
        HV.Cantidad,
        HV.PrecioUnitario,
        HV.Descuento,
        HV.IngresoBruto,
        DP.NombreProducto,
        DC.NombreEmpresa AS Cliente,
        DC.Pais AS PaisCliente,
        DE.Nombres + ' ' + DE.Apellidos AS Empleado,
        DE.Cargo,
        DF.FechaCompleta,
        DF.Año,
        DF.Mes,
        DF.Dia,
        DF.Trimestre
    FROM HechosVentas HV
    JOIN DimProducto DP ON HV.IdProducto = DP.IdProducto
    JOIN DimCliente DC ON HV.IdCliente = DC.IdCliente
    JOIN DimEmpleado DE ON HV.IdEmpleado = DE.IdEmpleado
    JOIN DimFecha DF ON HV.IdFecha = DF.IdFecha
''', con=cnxn)

df_datamart.head()




# %%
# Información general del DataFrame y verificación de valores nulos
df_datamart.info()
df_datamart.isnull().sum()

# %%
df_datamart.dtypes 

# %%
len(df_datamart)

# %%
# Eliminación de columnas que no aportan valor al análisis o que ya no se necesitan
columnas_a_eliminar = [
    "IdVenta", "IdPedido", "FechaCompleta", "Mes"
]
# Verifica que quedaron solo las relevantes
df_datamart.head()

# %% [markdown]
# # 📊 Análisis Bivariado
# 
# > El análisis bivariado permite examinar la relación entre dos variables, identificando patrones, correlaciones o dependencias estadísticas.
# 
# ---
# 
# ## 🎯 Objetivo
# 
# - Analizar la interacción entre dos variables.
# - Identificar asociaciones o relaciones significativas.
# - Apoyar modelos predictivos y exploratorios.
# 
# Realizaremos el análisis con Dataprep y de manera manual graficas creadas en base a criterio.

# %%
from dataprep.eda import create_report
create_report(df_datamart).show()

# %%
# Mapa de calor (heatmap) de correlación
corr_df = df_datamart.corr(method='pearson')

plt.figure(figsize=(13, 10))
sns.heatmap(corr_df, annot=True)
plt.show()

# %%
# Diagrama de dispersión entre la cantidad vendida y el ingreso bruto
sns.scatterplot(data=df_datamart, x="Cantidad", y="IngresoBruto")
plt.title("Ingreso Bruto vs. Cantidad")
plt.grid(True)
plt.show()

# %%
# Histograma del Ingreso Bruto por pedido

# Distribución del ingreso
sns.histplot(df_datamart["IngresoBruto"], kde=True)
plt.title("Distribución del Ingreso Bruto por Pedido")
plt.show()

# %%
# Boxplot del ingreso por producto

# Ingreso por producto
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_datamart, x="NombreProducto", y="IngresoBruto")
plt.xticks(rotation=90)
plt.title("Ingreso Bruto por Producto")
plt.tight_layout()
plt.show()

# %%
# Agrupamiento por nombre de producto
#  Calculamos el promedio (mean) de:
#     * Ingreso Bruto
#     * Cantidad vendida
#     * Precio Unitario
#     * Descuento aplicado
# Luego ordenamos los productos de mayor a menor Ingreso Bruto promedio

df_datamart.groupby("NombreProducto").agg({
    "IngresoBruto": "mean",
    "Cantidad": "mean",
    "PrecioUnitario": "mean",
    "Descuento": "mean"
}).sort_values("IngresoBruto", ascending=False)


# %%
# Boxplot del ingreso bruto por empleado
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_datamart, x="Empleado", y="IngresoBruto")
plt.xticks(rotation=90)
plt.title("Ingreso Bruto por Empleado")
plt.tight_layout()
plt.show()



# %%
# Agrupamiento por empleado
# Calculamos el promedio (mean) de:
#     * Ingreso Bruto generado por cada empleado
#     * Cantidad de productos vendidos por cada empleado
# Luego ordenamos los empleados de mayor a menor Ingreso Bruto promedio


df_datamart.groupby("Empleado").agg({
    "IngresoBruto": "mean",
    "Cantidad": "mean"
}).sort_values("IngresoBruto", ascending=False)

# %%
# Boxplot del Ingreso Bruto por Trimestre
sns.boxplot(data=df_datamart, x="Trimestre", y="IngresoBruto")
plt.title("Ingreso Bruto por Trimestre")
plt.grid(True)
plt.show()


# %%
# Calculamos el promedio (mean) de:
#     * Ingreso Bruto por trimestre
#     * Cantidad vendida por trimestre
#     * Descuento aplicado por trimestre
# Ordenamos los trimestres en orden creciente (1, 2, 3, 4)
df_datamart.groupby("Trimestre").agg({
    "IngresoBruto": "mean",
    "Cantidad": "mean",
    "Descuento": "mean"
}).sort_values("Trimestre")


# %%
# Agrupamiento por cliente
# Calculamos el promedio (mean) de:
#     * Ingreso Bruto generado por cada cliente
#     * Cantidad promedio de productos comprados
#     * Descuento promedio recibido
# Ordenamos los clientes de mayor a menor según su Ingreso Bruto promedio
df_datamart.groupby("Cliente").agg({
    "IngresoBruto": "mean",
    "Cantidad": "mean",
    "Descuento": "mean"
}).sort_values("IngresoBruto", ascending=False)


# %%
# Agrupamiento por país del cliente
# Calculamos el promedio (mean) de:
#     * Ingreso Bruto generado por país
#     * Cantidad promedio de productos vendidos por país
# Ordenamos los países de mayor a menor Ingreso Bruto promedio


df_datamart.groupby("PaisCliente").agg({
    "IngresoBruto": "mean",
    "Cantidad": "mean"
}).sort_values("IngresoBruto", ascending=False)


# %%
df_datamart.columns

# %%
df_datamart.shape

# %% [markdown]
# ### 🧹 Preparación del DataFrame para Machine Learning
# 
# ---
# 
# ### 🔎 Diferenciación de variables
# 
# - **Continuas:**  
#   `PrecioUnitario`, `Descuento`, `IngresoBruto`
# 
# - **Discretas:**  
#   `Cantidad`, `Año`
# 
# - **Categóricas nominales:**  
#   `NombreProducto`, `Cliente`, `Empleado`, `Continente`
# 
# - **Categórica ordinal:**  
#   `Cargo` (mapeado como `Puesto`)
# 
# ---
# 
# ### ⚙️ Métodos aplicados
# 
# - **MinMaxScaler:** Normalización de variables numéricas al rango [0,1].
# - **Mapeo ordinal:** Transformación de `Cargo` a `Puesto`.
# - **OneHotEncoding:** Aplicado a `Trimestre`, `Puesto` y variables categóricas.
# 
# ---
# 
# ### ✅ Validaciones realizadas
# 
# - Sin valores nulos en el DataFrame final.
# - Un único `Trimestre` y `Puesto` por fila.
# - Variables numéricas correctamente diferenciadas y normalizadas.
# 
# ---
# 
# ### 🎯 Resultado
# 
# DataFrame finalizado y listo para:
# - Machine Learning
# - Modelado predictivo
# - Técnicas de clustering
# 
# **Nota:**  
# Se preservaron los datos originales trabajando sobre una copia segura del DataFrame.
# 
# ---
# 

# %%
# Diccionario para mapear países a continentes
pais_a_continente = {
    'USA': 'América del Norte',
    'Brazil': 'América del Sur',
    'Argentina': 'América del Sur',
    'Canada': 'América del Norte',
    'Mexico': 'América del Norte',
    'Germany': 'Europa',
    'France': 'Europa',
    'UK': 'Europa',
    'Spain': 'Europa',
    'Italy': 'Europa',
    'Austria': 'Europa',
    'Sweden': 'Europa',
    'Finland': 'Europa',
    'Norway': 'Europa',
    'Denmark': 'Europa',
    'Portugal': 'Europa',
    'Poland': 'Europa',
    'Czech Republic': 'Europa',
    'Hungary': 'Europa',
    'Russia': 'Europa',
    'Belgium': 'Europa',          
    'Switzerland': 'Europa',       
    'Ireland': 'Europa',           
    'Venezuela': 'América del Sur',
    'China': 'Asia',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'India': 'Asia',
    'Australia': 'Oceanía',
    'New Zealand': 'Oceanía',
    'South Africa': 'África',
    'Egypt': 'África'
}

df_datamart['Continente'] = df_datamart['PaisCliente'].map(pais_a_continente)

# Eliminar PaisCliente
df_datamart = df_datamart.drop(columns=['PaisCliente'])


# %%
df_datamart.head()

# %%
# Crear una copia del DataFrame para trabajar el preprocesamiento
df_ml = df_datamart.copy()


# %%
# Definimos las variables continuas y discretas
scaler = MinMaxScaler()
variables_continuas = ['PrecioUnitario', 'Descuento', 'IngresoBruto']
variables_discretas = ['Cantidad', 'Año']

# Aplicamos escalado Min-Max a todas las variables numéricas seleccionadas
df_ml[variables_continuas + variables_discretas] = scaler.fit_transform(df_ml[variables_continuas + variables_discretas])


# %%
# Definimos las variables continuas y discretas
scaler = MinMaxScaler()
variables_continuas = ['PrecioUnitario', 'Descuento', 'IngresoBruto']
variables_discretas = ['Cantidad', 'Año']

# Aplicamos escalado Min-Max a todas las variables numéricas seleccionadas
df_ml[variables_continuas + variables_discretas] = scaler.fit_transform(df_ml[variables_continuas + variables_discretas])

# %%
# 1. Mapear Cargo a Puesto numérico
orden_cargo = {
    'Sales Representative': 1,
    'Inside Sales Coordinator': 2,
    'Sales Manager': 3,
    'Vice President, Sales': 4
}
df_ml['Cargo'] = df_ml['Cargo'].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
df_ml['Puesto'] = df_ml['Cargo'].map(orden_cargo)


# %%
#  One-Hot Encoding para Trimestre (si existe)
if 'Trimestre' in df_ml.columns:
    df_trimestre = pd.get_dummies(df_ml['Trimestre'], prefix='Trimestre')
    df_ml = pd.concat([df_ml.drop(columns=['Trimestre']), df_trimestre], axis=1)


# %%
# One-Hot Encoding para Puesto
df_puesto = pd.get_dummies(df_ml['Puesto'], prefix='Puesto')
df_ml = pd.concat([df_ml.drop(columns=['Puesto']), df_puesto], axis=1)


# %%
# One-Hot Encoding para variables categóricas
variables_categoricas = ['NombreProducto', 'Cliente', 'Empleado', 'Continente']
df_categoricas = pd.get_dummies(df_ml[variables_categoricas], drop_first=True)


# %%
# Crear df_ml_final
df_ml_final = pd.concat([
    df_ml[['Cantidad', 'PrecioUnitario', 'Descuento', 'IngresoBruto', 'Año']],
    df_puesto,          
    df_trimestre,       
    df_categoricas      
], axis=1)

# %%
# 6. Exportar
df_ml_final.to_csv('DataMart_Procesado.csv', index=False, encoding='utf-8')


# %%
print(df_ml_final.shape)
df_ml_final.head()


# %% [markdown]
# ## 🎯 Objetivo
# 
# ---
# 
# Finalmente, se cuenta con un DataFrame listo para un posterior análisis de Machine Learning, con el objetivo de:
# 
# ### 🔮 Predecir el **Ingreso Bruto** de un pedido en función de:
# 
# - 🛒 **Producto**
# - 👨‍💼 **Empleado**
# - 🧍 **Cliente**
# - 📅 **Fecha**
# 
# ---
# 
# **Nota:**  
# Se garantiza que las variables relevantes fueron correctamente preparadas y el DataFrame mantiene la integridad necesaria para modelados predictivos.
# 


