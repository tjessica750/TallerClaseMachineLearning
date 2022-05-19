
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

url = 'diabetes.csv'
data = pd.read_csv(url)



# Limpiado de la data.

rankGlucose = [-1, 39, 79, 119, 159, 199]
NameGlucose = ['1', '2', '3', '4','5']
data.Glucose = pd.cut(data.Glucose, rankGlucose, labels=NameGlucose)
rankDiabetesPedigreeFunction = [-1, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
NameDiabetesPedigreeFunction = ['1', '2', '3', '4','5','6','7','8','9','10','11']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rankDiabetesPedigreeFunction, labels=NameDiabetesPedigreeFunction)
rankAge = [20, 24, 29, 41, 81]
NameAge = ["1", "2", "3", "4"]
data.Age = pd.cut(data.Age, rankAge, labels=NameAge)



# 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning
# 2. Utilizar Cross validation para entrenar y probar los modelos con mínimo 10 splits

# creacion del modelo
x = np.array(data.drop(['Outcome'],1))
y = np.array(data.Outcome)
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
matriz_confusion_knn, fpr_knn, tpr_knn  = matrizConfusionAuc(model, x_test, y_test, y_pred)
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
matriz_confusion_ada, fpr_ada, tpr_ada  = matrizConfusionAuc(model, x_test, y_test, y_pred)
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
matriz_confusion_dc, fpr_dc, tpr_dc  = matrizConfusionAuc(model, x_test, y_test, y_pred)
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
matriz_confusion_NB, fpr_NB, tpr_NB  = matrizConfusionAuc(model, x_test, y_test, y_pred)
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


    