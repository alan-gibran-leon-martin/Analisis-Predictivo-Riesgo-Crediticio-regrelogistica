import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
import numpy as np

df= pd.read_csv ("C:\\Users\\LENOVO\\Desktop\\alan\\ejercicioshsbc\\train_u6lujuX_CVtuZ9i.csv ")

#Limpieza de datos
df= df.drop_duplicates()
#print(df.isnull().sum())
#print(df.info())
df= df.dropna(thresh=len(df.columns)-2)
df["Gender"]= df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"]= df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"]= df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"]= df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["LoanAmount"]= df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Loan_Amount_Term"]= df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
df["Credit_History"]= df["Credit_History"].fillna(df["Credit_History"].mean())
#print(df.isnull().sum())

#Transformar categorias
""""Se remplazaran las columnas categoricas "Gender", "Married","Self_Employed", "Education", "Property_Area"
 a valores nominales numéricos, puesto que carecen de orden lógico, y la columna Dependents  se respetara
  el orden modificando unicamente el valor de 3+ por 3 """
df2= pd.get_dummies(df, columns= ["Gender", "Married","Self_Employed", "Education", "Property_Area",
                                   "Loan_Status"], drop_first= True)
df2["Dependents"]= df2["Dependents"].replace("3+", "3")
columnas= ["Gender_Male", "Married_Yes","Self_Employed_Yes", "Education_Not Graduate", 
           "Property_Area_Semiurban", "Property_Area_Urban", "Loan_Status_Y","ApplicantIncome" ]
df2[columnas]= df2[columnas].astype(int)
df2["Dependents"]= df2["Dependents"].astype(int)
#print(df2["Loan_Status"].drop_duplicates())
#print(df2.info()) 

#Correlaciones
df2.drop(inplace= True, columns= ["Loan_ID"])
correlacion= df2.corr()

fig, ax = plt.subplots() #La variable con mas correlacion es Credit_History
mapa= ax.imshow(correlacion, cmap= "coolwarm", vmax= 1, vmin= -1)
ax.set_xticks(range(len(df2.columns)))
ax.set_yticks(range(len(df2.columns)))
ax.set_xticklabels(labels= df2.columns, rotation= 90)
ax.set_yticklabels(labels= df2.columns)
barra = ax.figure.colorbar(mapa, ax = ax)
barra.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
ax.set_title("Correlacion entre variables")
for x in range(len(correlacion.columns)):
    for y in range(len(correlacion.columns)):
        datos= correlacion.iloc[x,y]
        color= "black" if abs(datos)> 0.5  else  "white"
        ax.text(y,x, f'{datos:.1f}', ha= "center", va= "center", color= color)
plt.tight_layout()
#plt.show()

#Distribucion de las variables
columnas_num= df.select_dtypes(include=["int64", "float64"]).columns
df[columnas_num].hist(bins=30, grid= False)
plt.suptitle("Histograma de variables numéricas")
plt.tight_layout()
fig, ax3= plt.subplots(3,2)

ax3[0,0].boxplot(df["Credit_History"].dropna())
ax3[0,0].set_title("Credit_History")
ax3[0,1].boxplot(df["ApplicantIncome"].dropna())
ax3[0,1].set_title("ApplicantIncome")
ax3[1,0].boxplot(df["CoapplicantIncome"].dropna())
ax3[1,0].set_title("CoapplicantIncome")
ax3[1,1].boxplot(df["LoanAmount"].dropna())
ax3[1,1].set_title("LoanAmount")
ax3[2,0].boxplot(df["Loan_Amount_Term"].dropna())
ax3[2,0].set_title("Loan_Amount_Term")

fig.suptitle("Análisis boxplots")
plt.tight_layout()
#plt.show()

#Prueba con modelo Regresion Logistica en Credit_History
X= df2[["Credit_History"]] #Convierte en un dataframe en 2d
Y= df2["Loan_Status_Y"] #Es una serie 1d (en caso de que se convierta en 2d, usar ravel() en modelo.fit)
x_train, x_test, y_train, y_test = train_test_split(X, Y,train_size= .8, random_state=1234, 
                                                    shuffle= True) #solo usar suffle si no es serie temporal
modelo= LogisticRegression()
modelo.fit(x_train, y_train)
print("Intercepto: ", modelo.intercept_) #Intercepto, termino independiente o logaritmo de la probabilidad
                                        #indica la probabilidad de que sea aprobado el credito sin considerar
                                        #el historial crediticio o incluso si este fuera 0
print("Coeficiente: ", list(zip(X.columns, modelo.coef_.flatten(), )))#Sirve para saber que tanto afecta
                                      #la variable Credit_history a la aprobacion
print("Precisión: ", modelo.score(X,Y))# Que tanto predice correctamente el modelo (tener cuidado con sobreajuste)
prediccion= modelo.predict_proba(X= x_test) #Genera una tabla con probabilidades de que sea aprobado
                                        #o rechazado el credito
prediccion= pd.DataFrame(prediccion, columns= modelo.classes_)
#print(prediccion.head())

y_pred= modelo.predict(x_test)
y_prepro= modelo.predict_proba(x_test)

precision= accuracy_score (y_test, y_pred)
print("Precisión del test: ", precision) #Cuantos datos tuvo correctos

matriz_conf= confusion_matrix(y_test, y_pred)
print("Matriz de confusión \n", matriz_conf) # De izquierda a derecha positivos correctos (23), 
                                #falsos negativos (21), falsos positivos(2), negativos correcto(77).

reporte= classification_report(y_test, y_pred) 
print("Reporte de prediccion\n", reporte) #Primera columna precision el porcentaje true positive/ tp + fn, 
        #recall(sensibilidad) el porcentaje tp/ tp + fn , f1 es precision * recall / precision + recall
        #entre mas cercano a uno hay mayor equilibrio, support numero de muestras, accuracy porcentaje de
        #predicciones correctas, macro avg sirve para saber la importancia de cada clase, weighted avg
        #marca el rendimiento de modelos desbalanceados
"""En este caso hay una alta precisión para creditos rechazados (92%) y aprobados (79%), además de alta
sensibilidad en aprobado (97%), hay que tener cuidado ya que hay baja sensibilidad en rechazados( 52%) 
por lo que podrian haber una cantidad riesgosa de creditos que fueron aprobados incorrectamente, lo cual
podría llevar a impago
"""
#Curva ROC para ajustar umbral
y_proba= modelo.predict_proba(x_test)[:,1]
fpr, tpr, thresholds= roc_curve(y_test, y_proba) #fpr (falsos positivos rate)La proporción de casos 
                            #negativos reales (préstamos que deberían rechazarse) que el modelo 
                            #aprueba incorrectamente. 
optimal_idx= np.argmax(tpr-fpr)
umbral_optimo= thresholds[optimal_idx] 
print("Umbral optimo: ", umbral_optimo)

#Correccion con umbral optimo
y__corregida= (y_proba >= umbral_optimo).astype(int)
reporte2 = classification_report(y_test, y__corregida)
print("Reporte de prediccion 2: ", reporte2)
"""Al generar el nuevo reporte no se presentan diferencias en ninguna de las variables, por lo que se
asume que este modelo ha alcanzado su maxima predictivilidad coon la variable de historial crediticio.
Es adecuado para aprobar creditos de bajo riesgo, siendo la mayor debilidad predictiva la posibilidad
de aprobar creditos que deberían ser rechazados, pudiendo derivar en impago .Sera necesario revizar 
si es posible aumentar la sensibilidad con un modelo de regresión logística multiple """
print(df.columns)