# ==============================================================================
# 5.1 [CARGA DE DATOS DEL DATASET]
# ==============================================================================

# ----------------------------------------
# [Importar librerías]. ( import… )
# ----------------------------------------
print("--- 5.1 Importando Librerías ---")
import pandas as pd
import numpy as np
import scipy
from scipy import stats # Para análisis avanzado
import matplotlib.pyplot as plt
import seaborn as sns
print("Librerías importadas exitosamente.\n")

# ----------------------------------------
# [Lectura del Dataset]. ( read_json… )
# ----------------------------------------
print("--- 5.1 Leyendo el Dataset (JSON de datos.gov.co) ---")
# URL del dataset de Fondos de Inversión Colectiva (FIC)
url_datos = "https://www.datos.gov.co/resource/qhpu-8ixx.json"
# Aumentamos el límite de filas, ya que el dataset es grande
df = pd.read_json(f"{url_datos}?$limit=100000") # Leemos los últimos 100,000 registros
print(f"Dataset '{url_datos}' cargado exitosamente.\n")

# ----------------------------------------
# [Número de filas y columnas]. (shape…)
# ----------------------------------------
print("--- 5.1 Número de filas y columnas (shape) ---")
print(f"Dimensiones del dataset (muestra): {df.shape}")
print(f"Total de registros (muestra): {df.shape[0]}")
print(f"Total de variables: {df.shape[1]}\n")

# ----------------------------------------
# [Primeros 5 registros]. (head…)
# ----------------------------------------
print("--- 5.1 Primeros 5 registros (head) ---")
print(df.head())
print("\n")

# ----------------------------------------
# [Últimos 5 registros]. (tail…)
# ----------------------------------------
print("--- 5.1 Últimos 5 registros (tail) ---")
print(df.tail())
print("\n")

# ----------------------------------------
# [Tratamiento de Datos].
# ----------------------------------------
print("--- 5.1 Tratamiento de Datos ---")
# Lista de columnas que deben ser numéricas
cols_numericas = ['aportes_recibidos', 'retiros_redenciones',
                  'valor_unidad_operaciones', 'numero_inversionistas',
                  'rentabilidad_diaria', 'rentabilidad_mensual',
                  'rentabilidad_semestral', 'rentabilidad_anual',
                  'precierre_fondo_dia_t', 'valor_fondo_cierre_dia_t']

# Convertir columnas a tipo numérico, los errores se volverán NaN (Nulos)
for col in cols_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convertir 'fecha_corte' a formato de fecha
df['fecha_corte'] = pd.to_datetime(df['fecha_corte'])

# Manejar nulos (dropna): eliminamos filas donde las métricas clave no existen
df_tratado = df.dropna(subset=['aportes_recibidos', 'retiros_redenciones', 'rentabilidad_diaria'])

# Crear la columna Flujo_Neto (¡clave para el análisis!)
df_tratado['Flujo_Neto'] = df_tratado['aportes_recibidos'] - df_tratado['retiros_redenciones']

print("Datos tratados (tipos convertidos, nulos eliminados, 'Flujo_Neto' creado).\n")

# ----------------------------------------
# [Tipo de datos de las columnas]. (info…)
# ----------------------------------------
print("--- 5.1 Tipo de datos de las columnas (info) ---")
df_tratado.info()
print("\n")


# ==============================================================================
# 5.2 [VISUALIZACIÓN DE DATOS CON TABLAS]
# ==============================================================================

# ----------------------------------------
# [Visualización de una tabla por cada método de las librerías NumPy, SciPy y Pandas]
# ----------------------------------------
print("--- 5.2 Tabla con Pandas (DataFrame) ---")
print(df_tratado.head())
print("\n")

print("--- 5.2 Tabla con NumPy (Array) ---")
# Convertimos las primeras 5 filas de columnas numéricas a un array de NumPy
df_numpy = np.array(df_tratado[cols_numericas].head())
print(df_numpy)
print("\n")

print("--- 5.2 Tabla con SciPy (Matriz Dispersa) ---")
# Esto es solo un ejemplo, los datos de fondos no son dispersos
df_scipy = scipy.sparse.csr_matrix(df_tratado[cols_numericas].fillna(0).values)
print("Matriz dispersa de SciPy creada (solo se muestra la forma):")
print(df_scipy.shape)
print("\n")

# ----------------------------------------
# [Estadística descriptiva]. (describe)
# ----------------------------------------
print("--- 5.2 Estadística Descriptiva (describe) ---")
# Describimos las columnas numéricas clave
print(df_tratado[['rentabilidad_diaria', 'Flujo_Neto', 'numero_inversionistas']].describe())
print("\n")

# ----------------------------------------
# [Filtrar de valores]. (filter)
# ----------------------------------------
print("--- 5.2 Filtrar valores (días con rentabilidad extrema > 5%) ---")
df_filtrado = df_tratado[abs(df_tratado['rentabilidad_diaria']) > 0.05]
print(df_filtrado[['fecha_corte', 'nombre_patrimonio', 'rentabilidad_diaria']].head())
print("\n")

# ----------------------------------------
# [Datos agrupados]. (groupby)
# ----------------------------------------
print("--- 5.2 Datos agrupados (Flujo Neto total por subtipo de patrimonio) ---")
df_agrupado = df_tratado.groupby('nombre_subtipo_patrimonio')['Flujo_Neto'].sum().sort_values(ascending=False)
print(df_agrupado)
print("\n")

# ----------------------------------------
# [Datos ordenados]. (sort_values)
# ----------------------------------------
print("--- 5.2 Datos ordenados (Top 5 fondos con más inversionistas) ---")
df_ordenado = df_tratado.sort_values(by='numero_inversionistas', ascending=False)
print(df_ordenado[['nombre_patrimonio', 'nombre_entidad', 'numero_inversionistas']].head())
print("\n")

# ----------------------------------------
# [Análisis Básico].
# ----------------------------------------
print("--- 5.2 Análisis Básico (sobre la columna 'rentabilidad_diaria') ---")
col_analisis = 'rentabilidad_diaria'
print(f"Columna de análisis: {col_analisis}")

print(f"Índice dato máximo (idxmax): {df_tratado[col_analisis].idxmax()} (Es el índice de la fila con la max rentabilidad)")
print(f"Sumatoria (sum): {df_tratado[col_analisis].sum()}")
print(f"Valor mínimo (min): {df_tratado[col_analisis].min():.4f}")
print(f"Valor máximo (max): {df_tratado[col_analisis].max():.4f}")
print(f"Cuenta de datos (count): {df_tratado[col_analisis].count()}")
print(f"Media (mean): {df_tratado[col_analisis].mean():.6f} (La rentabilidad diaria promedio)")
print(f"Mediana (median): {df_tratado[col_analisis].median():.6f}")
print(f"Desviación Estándar (std): {df_tratado[col_analisis].std():.4f} (La volatilidad/riesgo diario)")
print(f"Varianza (var): {df_tratado[col_analisis].var():.6f}")
print("\nIntervalos de datos (quantile) - (Value at Risk - VaR):")
print(df_tratado[col_analisis].quantile([0.01, 0.05, 0.95, 0.99])) # 1% VaR es el cuantil 0.01
print("\n")

print("--- 5.2 Coeficiente de correlación (corr) ---")
# Correlación entre aportes y retiros
print(df_tratado[['aportes_recibidos', 'retiros_redenciones', 'Flujo_Neto']].corr())
print("\n")

# ----------------------------------------
# [Análisis avanzado] (Probabilidad y estadística con SciPy)
# ----------------------------------------
print("--- 5.2 Análisis Avanzado (SciPy) ---")
# 1. Prueba de Normalidad (Shapiro-Wilk)
# Hipótesis Nula: Los datos (rentabilidad) siguen una distribución normal.
# Usamos una muestra de 5000)
muestra_rentabilidad = df_tratado[col_analisis].sample(min(4999, len(df_tratado)))
shapiro_test = stats.shapiro(muestra_rentabilidad)
print(f"Prueba de Normalidad (Shapiro-Wilk) sobre '{col_analisis}':")
print(f"Estadístico={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue}")
if shapiro_test.pvalue < 0.05:
    print("Resultado: Las rentabilidades NO siguen una distribución normal (tienen 'colas pesadas').\n")
else:
    print("Resultado: Las rentabilidades parecen seguir una distribución normal.\n")

# 2. Prueba T de 1 muestra (One-sample T-test)
# Hipótesis Nula: La media de la rentabilidad diaria es igual a 0.
ttest_1samp = stats.ttest_1samp(df_tratado[col_analisis], 0)
print(f"Prueba T (media de '{col_analisis}' es 0):")
print(f"Estadístico T={ttest_1samp.statistic:.4f}, p-value={ttest_1samp.pvalue:.4f}")
if ttest_1samp.pvalue < 0.05:
    print("Resultado: La rentabilidad diaria promedio es estadísticamente DIFERENTE de 0.\n")
else:
    print("Resultado: No podemos rechazar que la rentabilidad diaria promedio sea 0.\n")


# ==============================================================================
# 5.3 [VISUALIZACIÓN DE DATOS CON GRÁFICAS]
# ==============================================================================
print("--- 5.3 Generando Gráficos (Matplotlib y Seaborn) ---")
print("Por favor, revisa las salidas de gráficos que se mostrarán a continuación.")

# ----------------------------------------
# [Representación gráfica con Matplotlib].
# ----------------------------------------

# [Diagramas de líneas]. (plot) - ¡El gráfico CLAVE de Flujo de Capital!
plt.figure(figsize=(12, 6))
# Agrupamos el Flujo Neto por día para ver el sentimiento del mercado
df_flujo_tiempo = df_tratado.groupby('fecha_corte')['Flujo_Neto'].sum()
# Calculamos una media móvil de 30 días para suavizar la tendencia
df_flujo_tiempo.rolling(window=30).mean().plot()
plt.title('Sentimiento del Mercado: Flujo Neto de Capital en FICs (SMA 30 Días)')
plt.xlabel('Fecha')
plt.ylabel('Flujo Neto de Capital (Aportes - Retiros)')
plt.grid(True)
plt.show()

# [Diagramas de barras]. (bar)
plt.figure(figsize=(10, 6))
# Usamos los datos agrupados que calculamos antes
df_agrupado.head(10).plot(kind='bar') # Top 10 subtipos
plt.title('Flujo Neto Total por Subtipo de Patrimonio (Top 10) (Matplotlib)')
plt.xlabel('Subtipo de Patrimonio')
plt.ylabel('Flujo Neto Total')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# [Histograma]. (hist)
plt.figure(figsize=(10, 6))
# Filtramos valores extremos para ver mejor la distribución
rentabilidad_filtrada = df_tratado['rentabilidad_diaria'].clip(-0.05, 0.05)
plt.hist(rentabilidad_filtrada, bins=100, edgecolor='black')
plt.title('Histograma de Rentabilidad Diaria (Matplotlib)')
plt.xlabel('Rentabilidad Diaria (limitada al 5%)')
plt.ylabel('Frecuencia')
plt.show()

# [Diagramas de dispersión]. (scatter)
plt.figure(figsize=(10, 6))
# Tomamos una muestra para que el gráfico sea legible
df_sample = df_tratado.sample(min(2000, len(df_tratado)))
plt.scatter(df_sample['aportes_recibidos'], df_sample['retiros_redenciones'], alpha=0.3)
plt.title('Correlación Aportes vs. Retiros (Matplotlib)')
plt.xlabel('Aportes Recibidos')
plt.ylabel('Retiros (Redenciones)')
plt.xscale('log') # Usamos escala logarítmica por los valores extremos
plt.yscale('log')
plt.show()

# ----------------------------------------
# [Representación gráfica con Seaborn].
# ----------------------------------------

# [Histograma]. (histplot)
plt.figure(figsize=(10, 6))
sns.histplot(rentabilidad_filtrada, bins=100, kde=True) # kde=True añade la curva
plt.title('Distribución de Rentabilidad Diaria (Seaborn)')
plt.xlabel('Rentabilidad Diaria (limitada al 5%)')
plt.ylabel('Frecuencia')
plt.show()

# [Gráfica de caja]. (boxplot)
plt.figure(figsize=(12, 7))
# Usamos los subtipos más comunes
top_subtipos = df_tratado['nombre_subtipo_patrimonio'].value_counts().nlargest(5).index
df_top_subtipos = df_tratado[df_tratado['nombre_subtipo_patrimonio'].isin(top_subtipos)]
# Filtramos rentabilidades extremas para ver las cajas
df_top_subtipos_filtrado = df_top_subtipos[df_top_subtipos['rentabilidad_diaria'].between(-0.05, 0.05)]

sns.boxplot(x='nombre_subtipo_patrimonio', y='rentabilidad_diaria', data=df_top_subtipos_filtrado)
plt.title('Boxplot de Rentabilidad por Subtipo de Patrimonio (Seaborn)')
plt.xlabel('Subtipo de Patrimonio')
plt.ylabel('Rentabilidad Diaria')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# [Gráfica de dispersión]. (scatterplot) - ¡Gráfico de 3 variables!
plt.figure(figsize=(12, 8))
# 3 variables: aportes (x), retiros (y), tipo de patrimonio (color)
sns.scatterplot(data=df_sample,
                x='aportes_recibidos',
                y='retiros_redenciones',
                hue='nombre_tipo_patrimonio', # 3ra variable
                alpha=0.5)
plt.title('Aportes vs. Retiros, coloreado por Tipo de Patrimonio (Seaborn)')
plt.xlabel('Aportes (Log)')
plt.ylabel('Retiros (Log)')
plt.xscale('log')
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()


print("\n--- ¡Análisis completo! ---")
