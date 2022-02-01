#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:55:31 2021

@author: juana
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

class CancerDiagnostic():

#%%
    def __init__(self, path, filename):
        """
        Default Constructor.

        Parameters
        ----------
        path : string
            CSV file path.
        filename : string
            CSV filename.

        Returns
        -------
        None.

        """
        self._path = path
        self._filename = filename
        self._diagnostic_data = pd.DataFrame()


#%%        
    def loadfile(self):
        """
        Load CSV file into dataframe.

        Returns
        -------
        None.

        """
        filename = os.path.join(self._path, self._filename)
        self._diagnostic_data = pd.read_csv(filename)

#%%
    def showGeneralInformation(self):
        """
        show the dataframe.

        Returns
        -------
        None.

        """
        print("General information of the dataset ".ljust(80, '-'), "\n")
        print(self._diagnostic_data.info())
        print("\n")

#%%        
    def deleteColumns(self):
        """
        Delete the unrelevant fields from the data set.

        Returns
        -------
        None.

        """
        self._diagnostic_data = self._diagnostic_data.drop(columns=['id'])

        self._diagnostic_data = self._diagnostic_data.drop(columns=['radius_se', 'texture_se'])
        self._diagnostic_data = self._diagnostic_data.drop(columns=['perimeter_se', 'area_se'])
        self._diagnostic_data = self._diagnostic_data.drop(columns=['smoothness_se', 'compactness_se'])
        self._diagnostic_data = self._diagnostic_data.drop(columns=['concavity_se', 'concave points_se'])
        self._diagnostic_data = self._diagnostic_data.drop(columns=['symmetry_se', 'fractal_dimension_se'])
        
        # self._diagnostic_data = self._diagnostic_data.drop(columns=['radius_worst', 'texture_worst'])
        # self._diagnostic_data = self._diagnostic_data.drop(columns=['perimeter_worst', 'area_worst'])
        # self._diagnostic_data = self._diagnostic_data.drop(columns=['smoothness_worst', 'compactness_worst'])
        # self._diagnostic_data = self._diagnostic_data.drop(columns=['concavity_worst', 'concave points_worst'])
        # self._diagnostic_data = self._diagnostic_data.drop(columns=['symmetry_worst', 'fractal_dimension_worst'])
        self._diagnostic_data = self._diagnostic_data.drop(columns=['Unnamed: 32'])

#%%        
    def showDF(self):
        """
        show the dataframe.

        Returns
        -------
        None.

        """
        print(self._diagnostic_data)
        print("\n")

#%%        
    def uniqueValues(self, column):
        """
        Get the unique values of the column.
        
        Parameters
        ----------
        
        column : int
            Diagnostic column.

        Returns
        -------
        Numpy array
            The unique values array.

        """
        return self._diagnostic_data[column].unique()

#%%        
    def replaceTags(self):
        """
        Replace the values of the diagnosis column to binary values.

        Returns
        -------
        None.

        """
        values_cross = {
            "diagnosis": {
                'M': 1,
                'B': 0
            }
        }

        self._diagnostic_data.replace(values_cross, inplace = True)

#%%        
    def toNumerical(self):
        """
        Convert to numerical values.

        Returns
        -------
        None.

        """
        self._diagnostic_data['diagnosis'] = pd.to_numeric(self._diagnostic_data['diagnosis'], errors='coerce')

#%%
    def checkNaN(self):
        """
        Check if exist NaN values.

        Returns
        -------
        TYPE
            True if a NaN value was found.

        """
        return self._diagnostic_data.isna().sum()
    
#%%
    def plotHist(self):
        """
        Plot using hist.

        Returns
        -------
        None.

        """
        self._diagnostic_data['radius_mean'].hist()
        #self._diagnostic_data['texture_mean'].hist()
        #self._diagnostic_data['perimeter_mean'].hist()
        #self._diagnostic_data['area_mean'].hist()
        #self._diagnostic_data['smoothness_mean'].hist()
        

#%%
    def plot(self):
        """
        Plot using scatter.

        Returns
        -------
        None.

        """
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['radius_mean'])
        plt.title('Diagnosis Vs Radius_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('radius_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['texture_mean'])
        plt.title('Diagnosis Vs Texture_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('texture_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['perimeter_mean'])
        plt.title('Diagnosis Vs Perimeter_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('perimeter_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['area_mean'])
        plt.title('Diagnosis Vs Area_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('area_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['smoothness_mean'])
        plt.title('Diagnosis Vs Smoothness_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('smoothness_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['compactness_mean'])
        plt.title('Diagnosis Vs Compactness_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('compactness_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['concavity_mean'])
        plt.title('Diagnosis Vs Concavity_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('concavity_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['concave points_mean'])
        plt.title('Diagnosis Vs Concave points_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('concave points_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['symmetry_mean'])
        plt.title('Diagnosis Vs Symmetry_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('symmetry_mean')
        plt.show()
        
        plt.scatter(x = self._diagnostic_data['diagnosis'], y = self._diagnostic_data['fractal_dimension_mean'])
        plt.title('Diagnosis Vs Fractal_dimension_mean')
        plt.xlabel('diagnosis')
        plt.ylabel('fractal_dimension_mean')
        plt.show()
        
        plt.close()
        
#%%
    def plotUsingSNS(self, column):
        """
        Plot using SNS.

        Parameters
        ----------
        column : int
            Diagnostic column.

        Returns
        -------
        None.

        """
        sns.catplot(
            x=column, 
            data=self._diagnostic_data, 
            kind="count"
        )
        plt.close()
        
#%%
    def countData(self, column):
        """
        Count the number of times the diagnostic values are repeated.

        Parameters
        ----------
        column : int
            Diagnostic column.

        Returns
        -------
        None.

        """
        result = dict(Counter(self._diagnostic_data[column]))
        
        print(''.ljust(30, '-'))
        
        for k, v in result.items():
            if k == 0:
                output = '| Benign: {:4} | TOTAL: {:4} |'.format(k, v)
            else:
                output = '| Malignant: {:1} | TOTAL: {:4} |'.format(k, v)
            print(output)
            
        print(''.ljust(30, '-'))
    
#%%
    def splitData(self, column):
        """
        Split data to get the training data.

        Parameters
        ----------
        column : int
            Diagnostic column.

        Returns
        -------
        None.

        """
        x = self._diagnostic_data.drop(columns = column)
        y = self._diagnostic_data[column]
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size = 0.2, random_state = 0
        )
         
#%%
    def applyRegression(self):
        """
        Apply logistic regression model.

        Returns
        -------
        None.

        """
        warnings.filterwarnings('ignore')
        self._model = LogisticRegression(
            solver='lbfgs', max_iter=1000
        )
        self._model.fit(self._x_train, self._y_train)

#%%
    def prediction(self):
        """
        Get the predictions with the test value.

        Returns
        -------
        None.

        """
        self._y_pred = self._model.predict(self._x_test)

#%%        
    def confusionMatrix(self):
        """
        Show confusion matrix comparing test and predictions.

        Returns
        -------
        None.

        """
        self._cm = confusion_matrix(self._y_test, self._y_pred)
        
#%%
    def plotMatrixPrettier(self):
        """
        Show confusion matrix with sns comparing test and predictions.

        Returns
        -------
        None.

        """
        group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        group_counts = ["{0:0.0f}".format(value) for value in self._cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in self._cm.flatten() / np.sum(self._cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(self._cm, annot=labels, fmt='', cmap='Blues')
        
        plt.close()
        
#%%        
    def showScore(self):
        """
        Show the score.

        Returns
        -------
        None.

        """
        print(
            "Model score: ", 
            self._model.score(
                self._x_train, self._y_train
            ) * 100
        )

#%%        
    def showAccuracy(self):
        """
        Show the accuracy.

        Returns
        -------
        None.

        """
        print(
            "Model acurracy: ", 
            accuracy_score(
                self._y_test, self._y_pred
            ) * 100
        )
        
#%%        
    def decisionTree(self):
        """
        Get Decision Tree Classifier.

        Returns
        -------
        None.

        """
        self._dTree = DecisionTreeClassifier()
        self._dTree.fit(self._x_train, self._y_train)
        
#%%        
    def decisionTreePrediction(self):
        """
        Decision tree prediction.

        Returns
        -------
        None.

        """
        self._y_pred = self._dTree.predict(self._x_test)    
        
#%%        
    def showReport(self):
        """
        Show report.

        Returns
        -------
        None.

        """
        print(classification_report(self._y_test, self._y_pred))
        
#%%        
    def showScoreDT(self):
        """
        Show the score for Decision Tree Classifier.

        Returns
        -------
        None.

        """
        print(
            "Model score: ", 
            self._dTree.score(
                self._x_train, self._y_train
            ) * 100
        )   
        
#%%
    def plotMatrixPrettierDT(self):
        """
        Show confusion matrix with sns comparing test and predictions.

        Returns
        -------
        None.

        """
        group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        group_counts = ["{0:0.0f}".format(value) for value in self._cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in self._cm.flatten() / np.sum(self._cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(self._cm, annot=labels, fmt='', cmap='Blues')
        
# End Class      
 
#%%
def prepareData():
    """
    Prepare the data.

    Returns
    -------
    None.

    """
    print("1. Prepare data ".ljust(80, '-'), "\n")
    
    #1.1 Load data.
    print("1.1 Load data ".ljust(80, '-'))
    cancer.loadfile()
    print("\n")
    
    cancer.showGeneralInformation()
    
    cancer.plotHist()
    
    #1.2 Choose the most relevant fields from the data set.
    #Delete the columns.
    print("1.2 Choose the most relevant fields from the data set ".ljust(80, '-'), "\n")
    cancer.deleteColumns()
    print("\n")
    
    cancer.showDF()
    
    cancer.showGeneralInformation()
    
    print("1.3 Check non-numerical values".ljust(80, '-'), "\n")

    #Check the different values that exist.
    print("1.3.1 Check the different values that exist".ljust(80, '-'), "\n")
    print("Diagnosis:")
    print(cancer.uniqueValues("diagnosis"))
    print("\n")
    
    #Assign a different value to each value.
    print("1.3.2 Assign a different value to each value:".ljust(80, '-'), "\n")
    print("Diagnosis:")
    cancer.replaceTags()
    cancer.showDF()
    
    #1.4 Converting an object to a numerical data (diagnosis).
    print("1.4 Converting an object to a numerical ".ljust(80, '-'), "\n")
    cancer.toNumerical()
    cancer.showGeneralInformation()
    
    #1.5 Check for NaN data and take action.
    print("1.5 Check for NaN data and take action ".ljust(80, '-'), "\n")
    print(cancer.checkNaN())
    print("\n")

"""
x_train : pandas.core.frame.DataFrame
    Get the data without diagnostic column.
x_test : pandas.core.frame.DataFrame
    Get the data without diagnostic column randomly by 20%.
y_train : pandas.core.frame.DataFrame
     Get the diagnostic column.
y_test : pandas.core.frame.DataFrame
    Get the diagnostic column randomly by 20.
"""

#%%    
def trainModel():
    """
    Train the model.

    Returns
    -------
    None.

    """
    print("2. Train the model ".ljust(80, '-'), "\n")
    
    #2.1 Graphics.
    print("2.1 Graphics: ".ljust(80, '-'), "\n")
    cancer.plot()
    
    cancer.plotUsingSNS('diagnosis')
    
    #2.2 Count the number of times the diagnostic values are repeated.
    print("2.2 Count diagnostic values that are repeated: ".ljust(80, '-'), "\n")
    cancer.countData('diagnosis')
    print("\n")
     
    #2.3 Split data.
    print("2.3 Split data: ".ljust(80, '-'), "\n")
    cancer.splitData('diagnosis')
    
    #print("x_train" + "\n")
    #print(x_train)
    #print("x_test" + "\n")
    #print(x_test)
    #print("y_train" + "\n")
    #print(y_train)
    #print("y_test" + "\n")
    #print(y_test)
    
    #2.4 Apply logistic regression model.
    print("2.4 Apply logistic regression model: ".ljust(80, '-'), "\n")
    cancer.applyRegression()
    print("\n")
    
    #2.5 Predictions.
    print("2.5 Predictions: ".ljust(80, '-'), "\n")
    cancer.prediction()
    print("\n")
    
    #2.6 Confusion matrix.
    print("2.6 Confusion matrix: ".ljust(80, '-'), "\n")
    cancer.confusionMatrix()
    print(cancer._cm)
    print("\n")
    
    #2.7 Plot confusion matrix.
    print("2.7 Plot confusion matrix: ".ljust(80, '-'), "\n")
    cancer.plotMatrixPrettier()
    
    #2.8 Show model score.
    print("2.8 Show model score: ".ljust(80, '-'), "\n")
    cancer.showScore()
    print("\n")
    
    #2.9 Show model acurracy.
    print("2.9 Show model acurracy: ".ljust(80, '-'), "\n")
    cancer.showAccuracy()
    print("\n")
    
    #2.10 Decision tree.
    print("2.10 Decision tree: ".ljust(80, '-'), "\n")
    cancer.decisionTree()
    print("\n")
    
    #2.11 Decision tree prediction.
    print("2.11 Decision tree prediction: ".ljust(80, '-'), "\n")
    cancer.decisionTreePrediction()
    print("\n")
    
    #2.12 Show report.
    print("2.12 Show report: ".ljust(80, '-'), "\n")
    cancer.showReport()
    print("\n")
    
    #2.13 Confusion matrix with decision tree.
    print("2.13 Confusion matrix with decision tree: ".ljust(80, '-'), "\n")
    cancer.confusionMatrix()
    print(cancer._cm)
    print("\n")
    
    #2.14 Show model score with decision tree.
    print("2.14 Show model score with decision tree: ".ljust(80, '-'), "\n")
    cancer.showScoreDT()
    print("\n")
    
    #2.15 Show model acurracy with decision tree.
    print("2.15 Show model acurracy with decision tree: ".ljust(80, '-'), "\n")
    cancer.showAccuracy()
    print("\n")

    #2.16 Plot confusion matrix with decision tree.
    print("2.16 Plot confusion matrix with decision tree: ".ljust(80, '-'), "\n")
    cancer.plotMatrixPrettierDT()

#%%
def testModel():
    pass

#%%    
if __name__ == "__main__":
    global cancer
    cancer = CancerDiagnostic(".", "data.csv")
    
    prepareData()
    trainModel()
    testModel()
    