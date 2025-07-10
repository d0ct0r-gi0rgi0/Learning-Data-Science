# Librerías y lectura de datos:
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


os.chdir(r'C:\Users\jorge\Desktop\Máster\Módulo 7 - Minería de datos y modelización predictiva\Parte 1\Tarea')

from FuncionesMineria import ( lm, Rsq, validacion_cruzada_lm, modelEffectSizes, crear_data_modelo, Transf_Auto)


with open('datosTarea.pickle', 'rb') as f:
    datos = pickle.load(f)




###############################################################################
# ASIGNACIÓN DE VARIABLES.
###############################################################################

varObjCont = datos['Izda_Pct']
varObjBin = datos['Izquierda']
datos_input = datos.drop(['Izda_Pct', 'Izquierda'], axis = 1)

numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Mejores transformaciones para las variables numericas con respesto a los dos tipos de variables:
input_cont = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjCont)], axis = 1)
input_bin = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjBin)], axis = 1)

# Conjuntos de datos que contengan las variables explicativas y una de las variables objetivo:
todo_cont = pd.concat([input_cont, varObjCont], axis = 1)
todo_bin = pd.concat([input_bin, varObjBin], axis = 1)
with open('VinoTodo_bin.pickle', 'wb') as archivo:
    pickle.dump(todo_bin, archivo)
with open('VinoTodo_cont.pickle', 'wb') as archivo:
    pickle.dump(todo_cont, archivo)



variables = list(datos_input.columns)
var_numericas = datos_input.select_dtypes(include=['int', 'float']).columns.tolist()
var_categoricas = [var for var in variables if var not in var_numericas]

# Variables predictoras (X) y objetivo (Y):
X = datos[variables]
Y = datos[['Izda_Pct', 'Izquierda']]


# División de los datos en conjunto de entrenamiento y conjunto de prueba:
X_train, X_test, Y_train, Y_test = train_test_split(datos_input, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)




###############################################################################
# MODELO 1.
###############################################################################
var_cont_1 = datos[var_numericas].columns.tolist()
var_categ_1 = datos[var_categoricas].columns.tolist()
var_interac = []
modelo1 = lm(Y_train, X_train, var_cont_1, var_categ_1)
modelo1['Modelo'].summary()
r_train_1 = Rsq(modelo1['Modelo'], Y_train, modelo1['X'])
datos_nuevos_1 = crear_data_modelo(X_test, var_cont_1, var_categ_1, var_interac, original_data = datos_input)
modelEffectSizes(modelo1, Y_train, X_train, var_cont_1, var_categ_1)
r_test_1 = Rsq(modelo1['Modelo'], Y_test, datos_nuevos_1)
p1 = len(modelo1['Modelo'].params)






###############################################################################
# MODELO 2.
###############################################################################
var_cont_2 = []
var_categ_2 = ['CCAA', 'ActividadPpal', 'Densidad']
modelo2 = lm(Y_train, X_train, var_cont_2, var_categ_2)
modelEffectSizes(modelo2, Y_train, X_train, var_cont_2, var_categ_2)
modelo2['Modelo'].summary()
r_train_2 = Rsq(modelo2['Modelo'], Y_train, modelo2['X'])

datos_nuevos_2 = crear_data_modelo(X_test, var_cont_2, var_categ_2, original_data = datos_input)

r_test_2 = Rsq(modelo2['Modelo'], Y_test, datos_nuevos_2)
p2 = len(modelo2['Modelo'].params)




###############################################################################
# MODELO 3.
###############################################################################
var_cont_3 = []
var_categ_3 = ['ActividadPpal', 'Densidad', 'prop_missings']
modelo3 = lm(Y_train, X_train, var_cont_3, var_categ_3)
modelo3['Modelo'].summary()
r_train_3 = Rsq(modelo3['Modelo'], Y_train, modelo3['X'])
datos_nuevos_3 = crear_data_modelo(X_test, var_cont_3, var_categ_3, original_data = datos_input)
r_test_3 = Rsq(modelo3['Modelo'], Y_test, datos_nuevos_3)
p3 = len(modelo3['Modelo'].params)





###############################################################################
# MODELO 4.
###############################################################################
var_cont_4 = []
var_categ_4 = ['CCAA', 'ActividadPpal', 'Densidad']
var_interac_4 = [('CCAA', 'Densidad')]
modelo4 = lm(Y_train, X_train, var_cont_4, var_categ_4, var_interac_4)
modelo4['Modelo'].summary()
r_train_4 = Rsq(modelo4['Modelo'], Y_train, modelo4['X'])
datos_nuevos_4 = crear_data_modelo(X_test, var_cont_4, var_categ_4, var_interac_4, original_data = datos_input)
r_test_4 = Rsq(modelo4['Modelo'], Y_test, datos_nuevos_4)
p4 = len(modelo4['Modelo'].params)



data_r = {'Modelo':
              ['1', '2', '3', '4'],
          'R2_train':
              [r_train_1, r_train_2, r_train_3, r_train_4],
          'R2_test':
              [r_test_1, r_test_2, r_test_3, r_test_4],
          'Parámetros': [p1, p2, p3, p4]}
modelos = pd.DataFrame(data_r)


























###############################################################################
# VALIDACIÓN CRUZADA.
###############################################################################

# Vlidacion cruzada repetida para ver que modelo es mejor:
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})

for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas:
    modelo1VC = validacion_cruzada_lm(5, X_train, Y_train, var_cont_1, var_categ_1)
    modelo2VC = validacion_cruzada_lm(5, X_train, Y_train, var_cont_2, var_categ_2)
    modelo3VC = validacion_cruzada_lm(5, X_train, Y_train, var_cont_3, var_categ_3)
    modelo4VC = validacion_cruzada_lm(5, X_train, Y_train, var_cont_4, var_categ_4, var_interac_4)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'Rsquared': modelo1VC + modelo2VC + modelo3VC + modelo4VC,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 4,
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
    })
    
    # Concatena los resultados de esta repetición al DataFrame principal 'results'
    results = pd.concat([results, results_rep], axis=0)

    
# Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de R-squared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico 
    

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
# Contar el número de parámetros en cada modelo
num_params = [len(modelo1['Modelo'].params), len(modelo2['Modelo'].params), 
             len(modelo3['Modelo'].params), len(modelo4['Modelo'].params)]

# Teniendo en cuenta el R2, la estabilidad y el numero de parametros, nos quedamos con el modelo3
# Vemos los coeficientes del modelo ganador
modelo3['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
Rsq(modelo1['Modelo'], Y_train, modelo1['X'])
Rsq(modelo1['Modelo'], Y_test, datos_nuevos_1)

# Vemos las variables mas importantes del modelo ganador
modelEffectSizes(modelo1, Y_train, X_train, var_cont_1, var_categ_1)



