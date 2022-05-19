
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

url = 'bank-full.csv'
data = pd.read_csv(url)



# Limpiado de la data.


rank = [16, 32, 48, 64, 80, 96]
Name = ['1', '2', '3', '4', '5' ]
data.age = pd.cut(data.age, rank, labels=Name)
data.marital.replace(['single','married','divorced'],[0,1,2], inplace=True)
data.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown','retired',
 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid','student']
 ,[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
data.education.replace(['unknown','primary','secondary','tertiary'],[0,1,2,3], inplace=True)
data.default.replace(['no','yes'],[0, 1], inplace=True)
data.housing.replace(['no','yes'],[0, 1], inplace=True)
data.loan.replace(['no','yes'],[0, 1], inplace=True)
data.drop(['contact'], axis=1, inplace = True ) 
data.month.replace(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
 ,[1,2,3,4,5,6,7,8,9,10,11,12],inplace = True)
data.poutcome.replace(['unknown','failure','success','other'],[0, 1, 2, 3], inplace=True) 
data.y.replace(['no','yes'],[0, 1], inplace=True)
# data limpia


# 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning
# 2. Utilizar Cross validation para entrenar y probar los modelos con mínimo 10 splits

# creacion del modelo
x = np.array(data.drop(['y'],1))
y = np.array(data.y)
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

    

