# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

os.chdir(r'C:\Users\jorge\Desktop\Máster\Módulo 7 - Minería de datos y modelización predictiva\Parte 1\Tarea')

from FuncionesMineria import (Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm, crear_data_modelo)

with open('datosTarea_cont.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identificación de variables:
varObjCont = todo['Izda_Pct']
todo = todo.drop('Izda_Pct', axis = 1)
var_cont = todo.select_dtypes(include = ['int', 'float']).columns.tolist()
var_cont_sin_transf = ['CodigoProvincia', 'Population', 'TotalCensus', 'Age_0-4_Ptge', 'Age_under19_Ptge',
                       'Age_19_65_pct', 'Age_over65_pct', 'WomanPopulationPtge', 'ForeignersPtge',
                       'SameComAutonPtge', 'SameComAutonDiffProvPtge', 'DifComAutonPtge', 'UnemployLess25_Ptge',
                       'Unemploy25_40_Ptge', 'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge', 'IndustryUnemploymentPtge',
                       'ConstructionUnemploymentPtge', 'ServicesUnemploymentPtge', 'totalEmpresas',
                       'Industria', 'Construccion', 'ComercTTEHosteleria', 'Servicios', 'inmuebles', 'Pob2010',
                       'SUPERFICIE', 'PobChange_pct', 'PersonasInmueble', 'Explotaciones']
var_categ = [var for var in todo.columns.tolist() if var not in var_cont]


x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)
















######################################################
# MODELO MANUAL.
######################################################
var_cont_2 = []
var_categ_2 = ['CCAA', 'ActividadPpal', 'Densidad']
modeloManual = lm(y_train, x_train, var_cont_2, var_categ_2)
modeloManual['Modelo'].summary()
r_train_manual = Rsq(modeloManual['Modelo'], y_train, modeloManual['X'])
datos_nuevos_2 = crear_data_modelo(x_test, var_cont_2, var_categ_2, original_data = todo)
r_test_2 = Rsq(modeloManual['Modelo'], y_test, datos_nuevos_2)
param_manual = len(modeloManual['Modelo'].params)





















######################################################
# MODELO STEPWISE AIC.
######################################################
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloStepAIC['Modelo'].summary()
r_train_stepAIC = Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'],
                                                original_data = todo)
r_test_stepAIC = Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
param_stepAIC = len(modeloStepAIC['Modelo'].params)

######################################################
# MODELO STEPWISE BIC.
######################################################
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
modeloStepBIC['Modelo'].summary()
r_train_stepBIC = Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'],
                                                original_data = todo)
r_test_stepBIC = Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)
param_stepBIC = len(modeloStepBIC['Modelo'].params)




######################################################
# MODELO BACKWARD AIC.
######################################################
modeloBackAIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()
r_train_backAIC = Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'],
                                                original_data = todo)
r_test_backAIC = Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)
param_backAIC = len(modeloBackAIC['Modelo'].params)


######################################################
# MODELO BACKWARD BIC.
######################################################
modeloBackBIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
modeloBackBIC['Modelo'].summary()
r_train_backBIC = Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'],
                                                original_data = todo)
r_test_backBIC = Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)
param_backBIC = len(modeloBackBIC['Modelo'].params)




######################################################
# MODELO FORWARD AIC.
######################################################
modeloForwAIC = lm_forward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloForwAIC['Modelo'].summary()
r_train_forwAIC = Rsq(modeloForwAIC['Modelo'], y_train, modeloForwAIC['X'])
x_test_modeloForwAIC = crear_data_modelo(x_test, modeloForwAIC['Variables']['cont'], 
                                                modeloForwAIC['Variables']['categ'], 
                                                modeloForwAIC['Variables']['inter'],
                                                original_data = todo)
r_test_forwAIC = Rsq(modeloForwAIC['Modelo'], y_test, x_test_modeloForwAIC)
param_forwAIC = len(modeloForwAIC['Modelo'].params)

######################################################
# MODELO FORWARD BIC.
######################################################
modeloForwBIC = lm_forward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
modeloForwBIC['Modelo'].summary()
r_train_forwBIC = Rsq(modeloForwBIC['Modelo'], y_train, modeloForwBIC['X'])
x_test_modeloForwBIC = crear_data_modelo(x_test, modeloForwBIC['Variables']['cont'], 
                                                modeloForwBIC['Variables']['categ'], 
                                                modeloForwBIC['Variables']['inter'],
                                                original_data = todo)
r_test_forwBIC = Rsq(modeloForwBIC['Modelo'], y_test, x_test_modeloForwBIC)
param_forwBIC = len(modeloForwBIC['Modelo'].params)





######################################################
# RESULTADOS PARA MODELOS SIN INTERACCIONES.
######################################################

data_r = {'Método':
              ['Backward', 'Backward', 'Forward', 'Forward', 'Stepwise', 'Stepwise'],
          'Métrica': 
              ['AIC', 'BIC', 'AIC', 'BIC', 'AIC', 'BIC'],
          'R2_train':
              [r_train_backAIC, r_train_backBIC, r_train_forwAIC, r_train_forwBIC, r_train_stepAIC, r_train_stepBIC],
          'R2_test':
              [r_test_backAIC, r_test_backBIC, r_test_forwAIC, r_test_forwBIC, r_test_stepAIC, r_test_stepBIC],
          'Parámetros': [param_backAIC, param_backBIC, param_forwAIC, param_forwBIC, param_stepAIC, param_stepBIC]}
modelos = pd.DataFrame(data_r)
















































######################################################
# MODELO STEPWISE AIC CON INTERACCIONES 2 A 2.
######################################################
interacciones = ['UnemployLess25_Ptge','SameComAutonDiffProvPtge','AgricultureUnemploymentPtge', 'CCAA']
interacciones_unicas = list(combinations(interacciones, 2))
  
modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'AIC')
modeloStepAIC_int['Modelo'].summary()
r_train_stepAIC_int = Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])

x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                    modeloStepAIC_int['Variables']['categ'], 
                                                    modeloStepAIC_int['Variables']['inter'],
                                                    original_data = todo)

r_test_stepAIC_int = Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)
param_stepAIC_int = len(modeloStepAIC_int['Modelo'].params)



######################################################
# MODELO STEPWISE BIC CON INTERACCIONES 2 A 2.
######################################################
modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'BIC')
modeloStepBIC_int['Modelo'].summary()
r_train_stepBIC_int = Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])

x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                    modeloStepBIC_int['Variables']['categ'], 
                                                    modeloStepBIC_int['Variables']['inter'],
                                                    original_data = todo)
r_test_stepBIC_int = Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)
param_stepBIC_int = len(modeloStepBIC_int['Modelo'].params)





######################################################
# MODELO BACKWARD AIC CON INTERACCIONES 2 A 2.
######################################################  
modeloBackAIC_int = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'AIC')
modeloBackAIC_int['Modelo'].summary()
r_train_backAIC_int = Rsq(modeloBackAIC_int['Modelo'], y_train, modeloBackAIC_int['X'])

x_test_modeloBackAIC_int = crear_data_modelo(x_test, modeloBackAIC_int['Variables']['cont'], 
                                                    modeloBackAIC_int['Variables']['categ'], 
                                                    modeloBackAIC_int['Variables']['inter'],
                                                    original_data = todo)

r_test_backAIC_int = Rsq(modeloBackAIC_int['Modelo'], y_test, x_test_modeloBackAIC_int)
param_backAIC_int = len(modeloBackAIC_int['Modelo'].params)



######################################################
# MODELO BACKWARD BIC CON INTERACCIONES 2 A 2.
######################################################
modeloBackBIC_int = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'BIC')
modeloBackBIC_int['Modelo'].summary()
r_train_backBIC_int = Rsq(modeloBackBIC_int['Modelo'], y_train, modeloBackBIC_int['X'])

x_test_modeloBackBIC_int = crear_data_modelo(x_test, modeloBackBIC_int['Variables']['cont'], 
                                                    modeloBackBIC_int['Variables']['categ'], 
                                                    modeloBackBIC_int['Variables']['inter'],
                                                    original_data = todo)

r_test_backBIC_int = Rsq(modeloBackBIC_int['Modelo'], y_test, x_test_modeloBackBIC_int)
param_backBIC_int = len(modeloBackBIC_int['Modelo'].params)




######################################################
# MODELO FORWARD AIC CON INTERACCIONES 2 A 2.
######################################################
modeloForwAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'AIC')
modeloForwAIC_int['Modelo'].summary()
r_train_forwAIC_int = Rsq(modeloForwAIC_int['Modelo'], y_train, modeloForwAIC_int['X'])

x_test_modeloForwAIC_int = crear_data_modelo(x_test, modeloForwAIC_int['Variables']['cont'], 
                                                    modeloForwAIC_int['Variables']['categ'], 
                                                    modeloForwAIC_int['Variables']['inter'],
                                                    original_data = todo)

r_test_forwAIC_int = Rsq(modeloForwAIC_int['Modelo'], y_test, x_test_modeloForwAIC_int)
param_forwAIC_int = len(modeloForwAIC_int['Modelo'].params)



######################################################
# MODELO FORWARD BIC CON INTERACCIONES 2 A 2.
######################################################
modeloForwBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, interacciones_unicas, 'BIC')
modeloForwBIC_int['Modelo'].summary()
r_train_forwBIC_int = Rsq(modeloForwBIC_int['Modelo'], y_train, modeloForwBIC_int['X'])

x_test_modeloForwBIC_int = crear_data_modelo(x_test, modeloForwBIC_int['Variables']['cont'], 
                                                    modeloForwBIC_int['Variables']['categ'], 
                                                    modeloForwBIC_int['Variables']['inter'],
                                                    original_data = todo)

r_test_forwBIC_int = Rsq(modeloForwBIC_int['Modelo'], y_test, x_test_modeloForwBIC_int)
param_forwBIC_int = len(modeloForwBIC_int['Modelo'].params)


######################################################
# RESULTADOS PARA MODELOS CON INTERACCIONES.
######################################################

data_r = {'Método':
              ['Backward', 'Backward', 'Forward', 'Forward', 'Stepwise', 'Stepwise'],
          'Métrica': 
              ['AIC', 'BIC', 'AIC', 'BIC', 'AIC', 'BIC'],
          'R2_train':
              [r_train_backAIC_int, r_train_backBIC_int, r_train_forwAIC_int, r_train_forwBIC_int, r_train_stepAIC_int, r_train_stepBIC_int],
          'R2_test':
              [r_test_backAIC_int, r_test_backBIC_int, r_test_forwAIC_int, r_test_forwBIC_int, r_test_stepAIC_int, r_test_stepBIC_int],
          'Parámetros': [param_backAIC_int, param_backBIC_int, param_forwAIC_int, param_forwBIC_int, param_stepAIC_int, param_stepBIC_int]}
modelos_int = pd.DataFrame(data_r)











































######################################################
# MODELO STEPWISE AIC CON TRANSFORMACIONES.
######################################################
interacciones = []
interacciones_unicas = list(combinations(interacciones, 2))
  
modeloStepAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloStepAIC_trans['Modelo'].summary()
r_train_stepAIC_trans = Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])

x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                    modeloStepAIC_trans['Variables']['categ'], 
                                                    modeloStepAIC_trans['Variables']['inter'],
                                                    original_data = todo)

r_test_stepAIC_trans = Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)
param_stepAIC_trans = len(modeloStepAIC_trans['Modelo'].params)



######################################################
# MODELO STEPWISE BIC CON TRANSFORMACIONES.
######################################################
modeloStepBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloStepBIC_trans['Modelo'].summary()
r_train_stepBIC_trans = Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])

x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                    modeloStepBIC_trans['Variables']['categ'], 
                                                    modeloStepBIC_trans['Variables']['inter'],
                                                    original_data = todo)
r_test_stepBIC_trans = Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)
param_stepBIC_trans = len(modeloStepBIC_trans['Modelo'].params)




######################################################
# MODELO BACKWARD AIC CON TRANSFORMACIONES.
######################################################  
modeloBackAIC_trans = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloBackAIC_trans['Modelo'].summary()
r_train_backAIC_trans = Rsq(modeloBackAIC_trans['Modelo'], y_train, modeloBackAIC_trans['X'])

x_test_modeloBackAIC_trans = crear_data_modelo(x_test, modeloBackAIC_trans['Variables']['cont'], 
                                                    modeloBackAIC_trans['Variables']['categ'], 
                                                    modeloBackAIC_trans['Variables']['inter'],
                                                    original_data = todo)

r_test_backAIC_trans = Rsq(modeloBackAIC_trans['Modelo'], y_test, x_test_modeloBackAIC_trans)
param_backAIC_trans = len(modeloBackAIC_trans['Modelo'].params)



######################################################
# MODELO BACKWARD BIC CON TRANSFORMACIONES.
######################################################
modeloBackBIC_trans = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloBackBIC_trans['Modelo'].summary()
r_train_backBIC_trans = Rsq(modeloBackBIC_trans['Modelo'], y_train, modeloBackBIC_trans['X'])

x_test_modeloBackBIC_trans = crear_data_modelo(x_test, modeloBackBIC_trans['Variables']['cont'], 
                                                    modeloBackBIC_trans['Variables']['categ'], 
                                                    modeloBackBIC_trans['Variables']['inter'],
                                                    original_data = todo)

r_test_backBIC_trans = Rsq(modeloBackBIC_trans['Modelo'], y_test, x_test_modeloBackBIC_trans)
param_backBIC_trans = len(modeloBackBIC_trans['Modelo'].params)




######################################################
# MODELO FORWARD AIC CON TRANSFORMACIONES.
######################################################
modeloForwAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloForwAIC_trans['Modelo'].summary()
r_train_forwAIC_trans = Rsq(modeloForwAIC_trans['Modelo'], y_train, modeloForwAIC_trans['X'])

x_test_modeloForwAIC_trans = crear_data_modelo(x_test, modeloForwAIC_trans['Variables']['cont'], 
                                                    modeloForwAIC_trans['Variables']['categ'], 
                                                    modeloForwAIC_trans['Variables']['inter'],
                                                    original_data = todo)

r_test_forwAIC_trans = Rsq(modeloForwAIC_trans['Modelo'], y_test, x_test_modeloForwAIC_trans)
param_forwAIC_trans = len(modeloForwAIC_trans['Modelo'].params)



######################################################
# MODELO FORWARD BIC CON TRANSFORMACIONES.
######################################################
modeloForwBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloForwBIC_trans['Modelo'].summary()
r_train_forwBIC_trans = Rsq(modeloForwBIC_trans['Modelo'], y_train, modeloForwBIC_trans['X'])

x_test_modeloForwBIC_trans = crear_data_modelo(x_test, modeloForwBIC_trans['Variables']['cont'], 
                                                    modeloForwBIC_trans['Variables']['categ'], 
                                                    modeloForwBIC_trans['Variables']['inter'],
                                                    original_data = todo)

r_test_forwBIC_trans = Rsq(modeloForwBIC_trans['Modelo'], y_test, x_test_modeloForwBIC_trans)
param_forwBIC_trans = len(modeloForwBIC_trans['Modelo'].params)


######################################################
# RESULTADOS PARA MODELOS CON TRANSFORMACIONES.
######################################################

data_r = {'Método':
              ['Backward', 'Backward', 'Forward', 'Forward', 'Stepwise', 'Stepwise'],
          'Métrica': 
              ['AIC', 'BIC', 'AIC', 'BIC', 'AIC', 'BIC'],
          'R2_train':
              [r_train_backAIC_trans, r_train_backBIC_trans, r_train_forwAIC_trans, r_train_forwBIC_trans, r_train_stepAIC_trans, r_train_stepBIC_trans],
          'R2_test':
              [r_test_backAIC_trans, r_test_backBIC_trans, r_test_forwAIC_trans, r_test_forwBIC_trans, r_test_stepAIC_trans, r_test_stepBIC_trans],
          'Parámetros': [param_backAIC_trans, param_backBIC_trans, param_forwAIC_trans, param_forwBIC_trans, param_stepAIC_trans, param_stepBIC_trans]}
modelos_trans = pd.DataFrame(data_r)
















































######################################################
# MODELO STEPWISE AIC COMPLETO.
######################################################
interacciones = ['UnemployLess25_Ptge','SameComAutonDiffProvPtge','AgricultureUnemploymentPtge', 'CCAA']
interacciones_unicas = list(combinations(interacciones, 2))
  
modeloStepAIC_completo = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloStepAIC_completo['Modelo'].summary()
r_train_stepAIC_completo = Rsq(modeloStepAIC_completo['Modelo'], y_train, modeloStepAIC_completo['X'])

x_test_modeloStepAIC_completo = crear_data_modelo(x_test, modeloStepAIC_completo['Variables']['cont'], 
                                                    modeloStepAIC_completo['Variables']['categ'], 
                                                    modeloStepAIC_completo['Variables']['inter'],
                                                    original_data = todo)

r_test_stepAIC_completo = Rsq(modeloStepAIC_completo['Modelo'], y_test, x_test_modeloStepAIC_completo)
param_stepAIC_completo = len(modeloStepAIC_completo['Modelo'].params)



######################################################
# MODELO STEPWISE BIC COMPLETO.
######################################################
modeloStepBIC_completo = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloStepBIC_completo['Modelo'].summary()
r_train_stepBIC_completo = Rsq(modeloStepBIC_completo['Modelo'], y_train, modeloStepBIC_completo['X'])

x_test_modeloStepBIC_completo = crear_data_modelo(x_test, modeloStepBIC_completo['Variables']['cont'], 
                                                    modeloStepBIC_completo['Variables']['categ'], 
                                                    modeloStepBIC_completo['Variables']['inter'],
                                                    original_data = todo)
r_test_stepBIC_completo = Rsq(modeloStepBIC_completo['Modelo'], y_test, x_test_modeloStepBIC_completo)
param_stepBIC_completo = len(modeloStepBIC_completo['Modelo'].params)




######################################################
# MODELO BACKWARD AIC COMPLETO.
######################################################  
modeloBackAIC_completo = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloBackAIC_completo['Modelo'].summary()
r_train_backAIC_completo = Rsq(modeloBackAIC_completo['Modelo'], y_train, modeloBackAIC_completo['X'])

x_test_modeloBackAIC_completo = crear_data_modelo(x_test, modeloBackAIC_completo['Variables']['cont'], 
                                                    modeloBackAIC_completo['Variables']['categ'], 
                                                    modeloBackAIC_completo['Variables']['inter'],
                                                    original_data = todo)

r_test_backAIC_completo = Rsq(modeloBackAIC_completo['Modelo'], y_test, x_test_modeloBackAIC_completo)
param_backAIC_completo = len(modeloBackAIC_completo['Modelo'].params)



######################################################
# MODELO BACKWARD BIC COMPLETO.
######################################################
modeloBackBIC_completo = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloBackBIC_completo['Modelo'].summary()
r_train_backBIC_completo = Rsq(modeloBackBIC_completo['Modelo'], y_train, modeloBackBIC_completo['X'])

x_test_modeloBackBIC_completo = crear_data_modelo(x_test, modeloBackBIC_completo['Variables']['cont'], 
                                                    modeloBackBIC_completo['Variables']['categ'], 
                                                    modeloBackBIC_completo['Variables']['inter'],
                                                    original_data = todo)

r_test_backBIC_completo = Rsq(modeloBackBIC_completo['Modelo'], y_test, x_test_modeloBackBIC_completo)
param_backBIC_completo = len(modeloBackBIC_completo['Modelo'].params)




######################################################
# MODELO FORWARD AIC COMPLETO.
######################################################
modeloForwAIC_completo = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloForwAIC_completo['Modelo'].summary()
r_train_forwAIC_completo = Rsq(modeloForwAIC_completo['Modelo'], y_train, modeloForwAIC_completo['X'])

x_test_modeloForwAIC_completo = crear_data_modelo(x_test, modeloForwAIC_completo['Variables']['cont'], 
                                                    modeloForwAIC_completo['Variables']['categ'], 
                                                    modeloForwAIC_completo['Variables']['inter'],
                                                    original_data = todo)

r_test_forwAIC_completo = Rsq(modeloForwAIC_completo['Modelo'], y_test, x_test_modeloForwAIC_completo)
param_forwAIC_completo = len(modeloForwAIC_completo['Modelo'].params)



######################################################
# MODELO FORWARD BIC COMPLETO.
######################################################
modeloForwBIC_completo = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloForwBIC_completo['Modelo'].summary()
r_train_forwBIC_completo = Rsq(modeloForwBIC_completo['Modelo'], y_train, modeloForwBIC_completo['X'])

x_test_modeloForwBIC_completo = crear_data_modelo(x_test, modeloForwBIC_completo['Variables']['cont'], 
                                                    modeloForwBIC_completo['Variables']['categ'], 
                                                    modeloForwBIC_completo['Variables']['inter'],
                                                    original_data = todo)

r_test_forwBIC_completo = Rsq(modeloForwBIC_completo['Modelo'], y_test, x_test_modeloForwBIC_completo)
param_forwBIC_completo = len(modeloForwBIC_completo['Modelo'].params)


######################################################
# RESULTADOS PARA MODELOS COMPLETO.
######################################################

data_r = {'Método':
              ['Backward', 'Backward', 'Forward', 'Forward', 'Stepwise', 'Stepwise'],
          'Métrica': 
              ['AIC', 'BIC', 'AIC', 'BIC', 'AIC', 'BIC'],
          'R2_train':
              [r_train_backAIC_completo, r_train_backBIC_completo, r_train_forwAIC_completo, r_train_forwBIC_completo, r_train_stepAIC_completo, r_train_stepBIC_completo],
          'R2_test':
              [r_test_backAIC_completo, r_test_backBIC_completo, r_test_forwAIC_completo, r_test_forwBIC_completo, r_test_stepAIC_completo, r_test_stepBIC_completo],
          'Parámetros': [param_backAIC_completo, param_backBIC_completo, param_forwAIC_completo, param_forwBIC_completo, param_stepAIC_completo, param_stepBIC_completo]}
modelos_completos = pd.DataFrame(data_r)































######################################################
# VALIDACIÓN CRUZADA.
######################################################

# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)

for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas

    modelo_manual = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloManual['Variables']['cont']
        , modeloManual['Variables']['categ']
    )
    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )
    modelo_stepAIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC_trans['Variables']['cont']
        , modeloStepAIC_trans['Variables']['categ']
    )
    modelo_stepBIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
    )
    modelo_stepBIC_transInt = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_completo['Variables']['cont']
        , modeloStepBIC_completo['Variables']['categ']
        , modeloStepBIC_completo['Variables']['inter']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_manual + modelo_stepBIC + modelo_stepBIC_int + modelo_stepBIC_trans + modelo_stepBIC_trans + modelo_stepBIC_transInt
        , 'Resample': ['Rep' + str((rep + 1))]*5*6 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    















######################################################
# BOXPLOT VALIDACIÓN CRUZADA.
######################################################


plt.figure(figsize=(10, 6))
plt.grid(True)
grupo_metrica = results.groupby('Modelo')['Rsquared']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
plt.xlabel('Modelo')
plt.xticks([1, 2, 3, 4, 5, 6], ['Manual', 'S-BIC', 'S-BIC-I', 'S-AIC-T', 'S-BIC-T' , 'S-BIC-C'])
plt.ylabel('$R^2$')
plt.title('Comparación de modelos')
plt.show()


media_r2 = results.groupby('Modelo')['Rsquared'].mean()
std_r2 = results.groupby('Modelo')['Rsquared'].std()
num_params = [len(modeloManual['Modelo'].params), len(modeloStepAIC['Modelo'].params), len(modeloStepBIC_int['Modelo'].params),
              len(modeloStepAIC_trans['Modelo'].params), len(modeloStepBIC_trans['Modelo'].params), 
              len(modeloStepBIC_completo['Modelo'].params)]



















######################################################
# SELECCIÓN ALEATORIA.
######################################################
# Almacenamiento de las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []}

for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las tres fórmulas más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]

# ============================================================================
# De las 30 repeticiones, las 3 que más se repiten son:
#   1)  Clasificacion', 'CalifProductor', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   2)  'CalifProductor', Alcohol, ('Densidad', 'Clasificacion'), ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   3) 'Clasificacion', 'CalifProductor', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta'), ('Acidez', 'pH')


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
        , modeloStepBIC_trans['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    modelo4 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_3['cont']
        , var_3['categ']
        , var_3['inter']
    )
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
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
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC_trans['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')), 
                 len(frec_ordenada['Formula'][2].split('+'))]

# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloStepBIC_trans

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)
