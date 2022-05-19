
"""
@author: Jessica Torres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
#Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#clasificadores
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve


# Quitar los waring
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

url = 'weatherAUS.csv'
data = pd.read_csv(url)

# Limpiado de la data.

data.drop('Date', axis = 1, inplace = True)
data["MinTemp"].fillna(round(data.MinTemp.mean(),1), inplace = True) 
data["MaxTemp"].fillna(round(data.MaxTemp.mean(),1), inplace = True) 
data["Rainfall"].fillna(round(data.Rainfall.mean(),1), inplace = True) 
data["Evaporation"].fillna(round(data.Evaporation.mean(),1), inplace = True) 
data["Sunshine"].fillna(round(data.Sunshine.mean(),1), inplace = True) 
data["WindGustSpeed"].fillna(round(data.WindGustSpeed.mean(),1), inplace = True) 
data["WindSpeed9am"].fillna(round(data.WindSpeed9am.mean(),1), inplace = True) 
data["WindSpeed3pm"].fillna(round(data.WindSpeed3pm.mean(),1), inplace = True) 
data["Humidity9am"].fillna(round(data.Humidity9am.mean(),1), inplace = True) 
data["Humidity3pm"].fillna(round(data.Humidity3pm.mean(),1), inplace = True) 
data["Pressure9am"].fillna(round(data.Pressure9am.mean(),1), inplace = True) 
data["Pressure3pm"].fillna(round(data.Pressure3pm.mean(),1), inplace = True) 
data["Cloud9am"].fillna(round(data.Cloud9am.mean(),1), inplace = True) 
data["Cloud3pm"].fillna(round(data.Cloud3pm.mean(),1), inplace = True) 
data["Temp9am"].fillna(round(data.Temp9am.mean(),1), inplace = True) 
data["Temp3pm"].fillna(round(data.Temp3pm.mean(),1), inplace = True) 
data.dropna(subset = ["WindGustDir"], inplace=True)
data.dropna(subset = ["WindDir9am"], inplace=True)
data.dropna(subset = ["WindDir3pm"], inplace=True)
data.dropna(subset = ["RainToday"], inplace=True)
data.Location.replace(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney',
       'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong',
       'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo',
       'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil',
       'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth',
       'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings',
       'Darwin', 'Katherine', 'Uluru'],
          [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,
           27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0], inplace=True)
data.WindGustDir.replace(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'],
          [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0], inplace=True)
data.WindDir9am.replace(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW', 'N', 'WSW','ESE', 'E', 'NW', 'WNW', 'NNE'],
                        [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0], inplace=True)
data.WindDir3pm.replace(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE'],
                        [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0], inplace=True)
data.RainToday.replace(['No', 'Yes'],[0.0,1.0], inplace=True)
data.RainTomorrow.replace(['No', 'Yes'],[0.0,1.0], inplace=True) 
data.Location = pd.cut(data.Location, [-10,0,10,20,30,40,50], 
                       labels = ['1','2','3','4','5','6'])
data.MinTemp = pd.cut(data.MinTemp, [-10,0,10,20,30,40], 
                      labels = ['1','2','3','4','5'])
data.MaxTemp = pd.cut(data.MaxTemp, [-10,0,10,20,30,40,50], 
                      labels = ['1','2','3','4','5','6'])
data.Rainfall = pd.cut(data.Rainfall, [-50,0,50,100,150,200,250,300,350,400], 
                       labels = ['1','2','3','4','5','6','7','8','9'])
data.Evaporation = pd.cut(data.Evaporation, [-20,0,20,40,60,80,100], 
                          labels = ['1','2','3','4','5','6'])
data.Sunshine = pd.cut(data.Sunshine, [-5,0,5,10,15], 
                       labels = ['1','2','3','4'])
data.WindGustSpeed = pd.cut(data.WindGustSpeed, [-20,0,20,40,60,80,100,120,140], 
                            labels = ['1','2','3','4','5','6','7','8'])
data.Humidity9am = pd.cut(data.Humidity9am, [-20,0,20,40,60,80,100,120], 
                          labels = ['1','2','3','4','5','6','7'])
data.Humidity3pm = pd.cut(data.Humidity3pm, [-20,0,20,40,60,80,100,120], 
                          labels = ['1','2','3','4','5','6','7'])
data.Pressure9am = pd.cut(data.Pressure9am, [960,980,1000,1020,1040,1060], 
                          labels = ['1','2','3','4','5'])
data.Pressure3pm = pd.cut(data.Pressure3pm, [960,980,1000,1020,1040], 
                          labels = ['1','2','3','4'])
data.Temp9am = pd.cut(data.Temp9am, [-10,0,10,20,30,40,50], 
                      labels = ['1','2','3','4','5','6'])
data.Temp3pm = pd.cut(data.Temp3pm, [-10,0,10,20,30,40,50], 
                      labels = ['1','2','3','4','5','6'])
data.RISK_MM = pd.cut(data.RISK_MM, [-50,0,50,100,150,200,250,300,350,400], 
                      labels = ['1','2','3','4','5','6','7','8','9'])
# data limpia

# 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning
# 2. Utilizar Cross validation para entrenar y probar los modelos con mínimo 10 splits

# creacion del modelo
x = np.array(data.drop('RainTomorrow', axis = 1))
y = np.array(data.RainTomorrow)
# 0 is not, 1 is yes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



def metricsTraining(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = [] 
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_train = accuracy_score(model.predict(x_train), y_train)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred, accuracy_train

def matrizConfusionAuc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    return matriz_confusion, fpr, tpr



def showMetrics(str_model, acc_validation, acc_test, y_test, y_pred, acc_train, precisionScore, acc_validationKN, acc_testKN, acc_trainKN, precisionScoreKN,  acc_validationBC, acc_testBC, acc_trainBC, precisionScoreBC, recallScoreBC,f1ScoreBC, acc_validationTC, acc_testTC, acc_trainTC, precisionScoreTC, recallScoreTC,f1ScoreTC, acc_validationNB, acc_testNB, acc_trainNB, precisionScoreNB, recallScoreNB,f1ScoreNB):
    TableMetrics = pd.DataFrame({'Metric': ['LOGISTIC REGRESSION','KNEIGHBORNS','ADABOOST CLASSIFIER','DECISION TREE','GaussianNB'],
                     'Training Acurancy':[round(acc_train,4),round(acc_trainKN,4),round(acc_trainBC,4),round(acc_trainTC,4),round(acc_trainNB,4)],
                     'Validation Accurancy':[round(acc_validation,4),round(acc_validationKN,4),round(acc_validationBC,4),round(acc_validationTC,4),round(acc_validationNB,4)],
                     'Test Accurancy':[round(acc_test,4),round(acc_testKN,4),round(acc_testBC,4),round(acc_testTC,4),round(acc_testNB,4)],
                     'Precision':[round(precisionScore,4),round(precisionScoreKN,4),round(precisionScoreBC,4),round(precisionScoreTC,4),round(precisionScoreNB,4)],
                     'Recall':[round(recallScore,4),round(recallScoreKN,4),round(recallScoreBC,4),round(recallScoreTC,4),round(recallScoreNB,4)],
                     'F1 Score':[round(f1Score,4),round(f1ScoreKN,4),round(f1ScoreBC,4),round(f1ScoreTC,4),round(f1ScoreNB,4)]})
    print("punto 3")
    

    

def showMatrizConfusion(nameModel,matrizConfusion,nameModelKN,matrizConfusionKN, nameModelBC,matrizConfusionBC, nameModelTC,matrizConfusionTC, nameModelNB,matrizConfusionNB):
    print('-' * 40 + '\n')
    print(str.upper(nameModel))
    tabla = pd.DataFrame(matrizConfusion, index =['Positivo 1 ','Negativo 0'], columns=['positivo 1 ', 'Negativo 0'])
    print(tabla)

    print('-' * 40 + '\n')
    print(str.upper(nameModelKN))
    tablaKN = pd.DataFrame(matrizConfusionKN, index =['Positivo 1 ','Negativo 0'], columns=['positivo 1 ', 'Negativo 0'])
    print(tablaKN)

    print('-' * 40 + '\n')
    print(str.upper(nameModelBC))
    tablaBC = pd.DataFrame(matrizConfusionBC, index =['Positivo 1 ','Negativo 0'], columns=['positivo 1 ', 'Negativo 0'])
    print(tablaBC)

    print('-' * 40 + '\n')
    print(str.upper(nameModelTC))
    tablaTC = pd.DataFrame(matrizConfusionTC, index =['Positivo 1 ','Negativo 0'], columns=['positivo 1 ', 'Negativo 0'])
    print(tablaTC)

    print('-' * 40 + '\n')
    print(str.upper(nameModelNB))
    tablaNB = pd.DataFrame(matrizConfusionNB, index =['Positivo 1 ','Negativo 0'], columns=['positivo 1 ', 'Negativo 0'])
    print(tablaNB)

    
def graficarMapHead():
    f, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    sns.heatmap(matriz_confusion_lg, ax=axes[0,0])
    sns.heatmap(matriz_confusion_dc, ax=axes[0,1])
    sns.heatmap(matriz_confusion_knn, ax=axes[0,2])
    sns.heatmap(matriz_confusion_ada, ax=axes[1,0])
    sns.heatmap(matriz_confusion_NB, ax=axes[1,1])
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()
    
    
# punto 6
def metrics(str_model, y_test, y_pred):
    print('-' * 50 + '\n')
    print('Punto 6')
    print(str.upper(str_model))
    print('\n')
    print(classification_report(y_test, y_pred))






def showRocCurveMatrix(fpr, tpr, matriz_confusion):
    sns.heatmap(matriz_confusion)
    plt.show()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()    





model = LogisticRegression()
model, acc_validation, acc_test, y_pred, acc_train =  metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_lg, fpr_lg, tpr_lg  = matrizConfusionAuc(model, x_test, y_test, y_pred)
precisionScore = precision_score(y_test, y_pred)
recallScore = recall_score(y_test, y_pred)
f1Score = f1_score(y_test,y_pred)
matrizConfusion = confusion_matrix(y_test, y_pred)
nameModel = 'Logistic Regression'
Metrics = classification_report(y_test, y_pred)
metrics('Logistic Regression', y_test, y_pred)


showRocCurveMatrix(fpr_lg, tpr_lg, matriz_confusion_lg)


model = KNeighborsClassifier(n_neighbors = 3)
model, acc_validationKN, acc_testKN, y_pred, acc_trainKN = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_knn, AUCKN, fpr_knn, tpr_knn  = matrizConfusionAuc(model, x_test, y_test, y_pred)
precisionScoreKN = precision_score(y_test, y_pred)
recallScoreKN = recall_score(y_test, y_pred)
f1ScoreKN = f1_score(y_test,y_pred)
matrizConfusionKN = confusion_matrix(y_test, y_pred)
nameModelKN = 'KNeighbors Classifier'
MetricsKN = classification_report(y_test, y_pred)
metrics('KNeighbors Classifier', y_test, y_pred)



showRocCurveMatrix(fpr_knn, tpr_knn, matriz_confusion_knn)



model = AdaBoostClassifier(n_estimators=10)
model, acc_validationBC, acc_testBC, y_pred, acc_trainBC = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_ada, AUCBC, fpr_ada, tpr_ada  = matrizConfusionAuc(model, x_test, y_test, y_pred)
precisionScoreBC = precision_score(y_test, y_pred)
recallScoreBC = recall_score(y_test, y_pred)
f1ScoreBC = f1_score(y_test,y_pred)
matrizConfusionBC = confusion_matrix(y_test, y_pred)
nameModelBC = 'AdaBoost Classifier'
MetricsBC = classification_report(y_test, y_pred)
metrics('AdaBoost Classifier', y_test, y_pred)



showRocCurveMatrix(fpr_ada, tpr_ada, matriz_confusion_ada)


model = DecisionTreeClassifier()
model, acc_validationTC, acc_testTC, y_pred, acc_trainTC = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_dc, AUCTC, fpr_dc, tpr_dc  = matrizConfusionAuc(model, x_test, y_test, y_pred)
precisionScoreTC = precision_score(y_test, y_pred)
recallScoreTC = recall_score(y_test, y_pred)
f1ScoreTC = f1_score(y_test,y_pred)
matrizConfusionTC = confusion_matrix(y_test, y_pred)
nameModelTC = 'Decision Tree Classifier'
MetricsTC = classification_report(y_test, y_pred)
metrics('Decision Tree Classifier', y_test, y_pred)



showRocCurveMatrix(fpr_dc, tpr_dc, matriz_confusion_dc)



model = GaussianNB()
model, acc_validationNB, acc_testNB, y_pred, acc_trainNB = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_NB, AUCNB, fpr_NB, tpr_NB  = matrizConfusionAuc(model, x_test, y_test, y_pred)
precisionScoreNB = precision_score(y_test, y_pred)
recallScoreNB = recall_score(y_test, y_pred)
f1ScoreNB = f1_score(y_test,y_pred)
matrizConfusionNB = confusion_matrix(y_test, y_pred)
nameModelNB = 'Gaussian Naive Bayes'
MetricsNB = classification_report(y_test, y_pred)
metrics('Gaussian Naive Bayes', y_test, y_pred)


showRocCurveMatrix(fpr_NB, tpr_NB, matriz_confusion_NB)


# punto 3
showMetrics('KNeighborns', acc_validation, acc_test, y_test, y_pred, acc_train, precisionScore, acc_validationKN, acc_testKN, acc_trainKN, precisionScoreKN, acc_validationBC, acc_testBC, acc_trainBC, precisionScoreBC, recallScoreBC,f1ScoreBC, acc_validationTC, acc_testTC, acc_trainTC, precisionScoreTC, recallScoreTC,f1ScoreTC, acc_validationNB, acc_testNB, acc_trainNB, precisionScoreNB, recallScoreNB,f1ScoreNB) 

#punto 5 
showMatrizConfusion(nameModel,matrizConfusion,nameModelKN,matrizConfusionKN, nameModelBC,matrizConfusionBC, nameModelTC,matrizConfusionTC, nameModelNB,matrizConfusionNB)

# punto 5
graficarMapHead()






