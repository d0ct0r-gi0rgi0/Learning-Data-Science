# Librerías y lectura de datos: 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
os.chdir(r'C:\Users\jorge\Desktop\Máster\Módulo 7 - Minería de datos y modelización predictiva\Parte 1\Tarea')

from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali)

datos = pd.read_excel('DatosEleccionesEspaña.xlsx')
datos = datos.drop(columns = ['AbstentionPtge', 'Dcha_Pct', 'Otros_Pct', 'AbstencionAlta', 'Derecha'])
datos.dtypes




###############################################################################
# DEPURADO DE LOS DATOS.
###############################################################################

datos['Izquierda'] = datos['Izquierda'].astype('str')

# Separación entre columnas numéricas y categóricas:
variables = datos.columns.tolist()
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float',
                    'float32', 'float64']).columns.tolist()
categoricas = [variable for variable in variables if variable not in numericas]

# Creación de tabla estadística.
descriptivos_num = datos.describe().T
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)







###############################################################################
# COLUMNAS CATEGÓRICAS.
###############################################################################

frecuencias = analizar_variables_categoricas(datos)
frecuencias

lista_de_frecuencias = list(frecuencias['Name']['n'])
n = lista_de_frecuencias.count(2)
# print(f"Hay {n} municipios con el mismo nombre.")

municipios_repetidos = list(frecuencias['Name'].index)[:n]
del n

for municipio in municipios_repetidos:
    CCAA = datos.loc[datos['Name'] == municipio]['CCAA'].tolist()
    if CCAA[0] == CCAA[1]:
        print(f"ERROR: el municipio {municipio} está repetido en {CCAA[0]}.")


# Cambio de '?' por nan en la columna 'Densidad'.
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)
analizar_variables_categoricas(datos)


# Conversión de valores no numéricos de columnas numéricas en NaN.
datos[numericas] = datos[numericas].apply(pd.to_numeric, errors='coerce')
cuentaDistintos(datos)






###############################################################################
# COLUMNAS NUMÉRICAS.
###############################################################################

datos[variables].isna().sum()

# Cambio de cadenas de caracteres 'nan' a NaN.
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

# Missings no declarados variables cuantitativas (-1, 99999)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# Valores fuera de rango. 'PobChange_pct' puede contener porcentajes negativos.
c_no_porcentajes = ['Name', 'CodigoProvincia', 'CCAA', 'Population',
                    'TotalCensus', 'Izquierda', 'totalEmpresas', 'Industria',
                    'Construccion', 'ComercTTEHosteleria', 'Servicios',
                    'ActividadPpal', 'inmuebles', 'Pob2010', 'SUPERFICIE',
                    'Densidad', 'PersonasInmueble', 'Explotaciones']
c_porcentajes = [c for c in variables if c not in c_no_porcentajes]

for var in c_porcentajes:
    if var != 'PobChange_pct':
        datos[var] = [x if 0 <= x <= 100 else np.nan for x in datos[var]]











    
###############################################################################
# ELECCIÓN DE VARIABLES OBJETIVO.
###############################################################################

# Se establece la columna 'Name' como índice del DataFrame:
datos = datos.set_index(datos['Name']).drop('Name', axis = 1)


# Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
varObjCont = datos['Izda_Pct']
varObjBin = datos['Izquierda']
datos_input = datos.drop(['Izda_Pct', 'Izquierda'], axis = 1)

# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]








###############################################################################
# VALORES ATÍPICOS.
###############################################################################

# Porcentaje de atipicos de cada variable. 

# Seleccionar las columnas numéricas en el DataFrame.
# Calcular la proporción de valores atípicos para cada columna numérica.
# Utilizando una función llamada 'atipicosAmissing':
#   - 'x' representa el nombre de cada columna numérica mientras se itera a través de 'numericas'.
#   - 'atipicosAmissing(datos_input[x])' es una llamada a una función que devuelve una dupla donde
#           el segundo elemento ([1]) es el númeron de valores atípicos.
#   - 'len(datos_input)' es el número total de filas en el DataFrame de entrada.
#   - La proporción de valores atípicos se calcula dividiendo la cantidad de valores atípicos por el
#           número total de filas

numAtipicos = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

# Se cambian los atípicos a missing:
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]





###############################################################################
# MISSINGS.
###############################################################################

# Mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
patron_perdidos(datos_input)

# Proporción de valores perdidos por cada variable.
prop_missingsVars = datos_input.isna().sum()/len(datos_input)
prop_missingsVars

# Número de valores perdidos por cada observación:
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Estudio descriptivo básico a la nueva variable.
datos_input['prop_missings'].describe()

# Número de valores distintos que tiene la nueva variable.
len(datos_input['prop_missings'].unique())

# Se eliminan las observaciones con más del 50% de datos missing.
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# Transformación de la nueva variable a categórica:
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)


# Se eliminan las variables con más del 50% de datos missing.
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)

# Se crean otra vez las listas de variables por si acaso:
variables_input = list(datos_input.columns)
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]


# Tan sólo las columnas 'Population' y 'TotalCensus' tienen una proporción de valores missing superior al 9%.
# Por tanto, no se corrigen.
# datos_input['Population'] = datos_input['Population'].fillna('Desconocido')
# datos_input['TotalCensus'] = datos_input['TotalCensus'].fillna('Desconocido')








###############################################################################
# IMPUTACIONES.
###############################################################################

# Imputación aleatoria de variables cuantitativas:
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')

# Imputación aleatoria de variables cualitativas:
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Revisión de valores missing.
datos_input.isna().sum()


# Datos depurados guardados:
datosTarea = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosTarea.pickle', 'wb') as archivo:
    pickle.dump(datosTarea, archivo)






































