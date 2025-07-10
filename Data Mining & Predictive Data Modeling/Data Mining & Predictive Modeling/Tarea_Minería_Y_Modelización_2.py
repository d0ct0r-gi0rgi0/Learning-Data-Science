# Librerías y lectura de datos:
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir(r'C:\Users\jorge\Desktop\Máster\Módulo 7 - Minería de datos y modelización predictiva\Parte 1\Tarea')

from FuncionesMineria import (graficoVcramer, Vcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, crear_data_modelo, Vcramer, hist_target_categorica)

with open('datosTarea.pickle', 'rb') as f:
    datos = pickle.load(f)


###############################################################################
# VARIABLES OBJETIVO.
###############################################################################

varObjCont = datos['Izda_Pct']
varObjBin = datos['Izquierda']
datos_input = datos.drop(['Izda_Pct', 'Izquierda'], axis = 1)
variables = list(datos_input.columns)  



# Gráficos comparando los V de Cramer.
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)


# Cálculo de los coeficientes V de Cramer:
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)



###############################################################################
# GRÁFICOS BINARIA.
###############################################################################


# Las variables cualitativas con menos y más efecto sobre la variable objetivo binaria
# respectivamente son Industria y ActividadPpal.
mosaico_targetbinaria(datos_input['ActividadPpal'], varObjBin, 'ActividadPpal')
mosaico_targetbinaria(datos_input['CCAA'], varObjBin, 'CCAA')


# Boxplot: relación entre variables cuantitavivas y variable objetvo binaria.
boxplot_targetbinaria(datos_input['Servicios'], varObjBin, nombre_ejeX='Compra', nombre_ejeY='Servicios')
boxplot_targetbinaria(datos_input['UnemployLess25_Ptge'], varObjBin, nombre_ejeX='Compra', nombre_ejeY='UnemployLess25_Ptge')


# Histogramas:
hist_targetbinaria(datos_input['Servicios'], varObjBin, 'Servicios')
hist_targetbinaria(datos_input['UnemployLess25_Ptge'], varObjBin, 'UnemployLess25_Ptge')




###############################################################################
# GRÁFICOS CONTINUA.
###############################################################################


# Boxplot e histogramas para la variable objetivo continua:
    
boxplot_targetbinaria(varObjCont, datos_input['ActividadPpal'], nombre_ejeX = 'ActividadPpal', nombre_ejeY = varObjCont.name)
boxplot_targetbinaria(varObjCont, datos_input['CCAA'], nombre_ejeX = 'CCAA', nombre_ejeY = varObjCont.name)

hist_target_categorica(varObjCont, datos_input['Densidad'], nombre_ejeX = varObjCont.name, nombre_ejeY='Densidad')
hist_target_categorica(varObjCont, datos_input['CCAA'], nombre_ejeX = varObjCont.name, nombre_ejeY = 'CCAA')



###############################################################################
# MATRIZ DE CORRELACIÓN..
###############################################################################


# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(8, 6))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación de valores ausentes")
# Mostrar el gráfico de la matriz de correlación
plt.show()



