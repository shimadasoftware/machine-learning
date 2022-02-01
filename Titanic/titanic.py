#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:20:30 2021

@author: juana
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class Titanic():

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
        self._titanic_data = pd.DataFrame()
        
#%%        
    def loadfile(self):
        """
        Load CSV file into dataframe.

        Returns
        -------
        None.

        """
        filename = os.path.join(self._path, self._filename)
        self._titanic_data = pd.read_csv(filename)
        
#%%
    def showGeneralInformation(self):
        """
        Show the dataframe.

        Returns
        -------
        None.

        """
        print("General information of the dataset ".ljust(80, '-'), "\n")
        print(self._titanic_data.info())
        print("\n")

#%%        
    def showDF(self):
        """
        Show the dataframe.

        Returns
        -------
        None.

        """
        print("Show the dataframe ".ljust(80, '-'), "\n")
        print(self._titanic_data)
        print("\n")
        
#%%        
    def showDescription(self):
        """
        Show the dataframe.

        Returns
        -------
        None.

        """
        print("Show description of the entire fields".ljust(80, '-'), "\n")
        print(self._titanic_data.describe())
        print("\n")     
    
#%%
    def checkNaN(self):
        """
        Check if exist NaN values.

        Returns
        -------
        TYPE
            True if a NaN value was found.

        """
        return self._titanic_data.isna().sum()    
    
#%%
    def checkRelevantInfo(self):
        """
        Check if exist NaN values.

        Returns
        -------
        TYPE
            True if a NaN value was found.

        """
        print("Ckeck relevant information ".ljust(80, '-'), "\n")
        print(self._titanic_data.head())
        print("\n") 

#%%        
    def uniqueValues(self, column):
        """
        Get the unique values of the column.
        
        Parameters
        ----------
        
        column : int
            Species column.

        Returns
        -------
        Numpy array
            The unique values array.

        """
        return self._titanic_data[column].unique()    

#%%        
    def replaceTags(self, column):
        """
        Replace the values of the column to numerical values.

        Returns
        -------
        None.

        """
        if column == 'Sex':
            values_cross = {
                "Sex": {
                    'male': 1,
                    'female': 2,
                }
            }
        elif column == 'Embarked':
            values_cross = {
                "Embarked": {
                    'S': 1,
                    'C': 2,
                    'Q': 3
                }
            }

        self._titanic_data.replace(values_cross, inplace = True)

#%%        
    def toNumerical(self, param):
        """
        Convert to numerical values.

        Returns
        -------
        None.

        """
        self._titanic_data[param] = pd.to_numeric(self._titanic_data[param], errors='coerce')
        
#%%        
    def roundAge(self, param):
        """
        Convert to numerical values.

        Returns
        -------
        None.

        """
        print(self._titanic_data[param].mean())
        self._titanic_data[param] = self._titanic_data[param].replace(np.nan, 30)
        
#%%        
    def deleteColumns(self):
        """
        Delete the unrelevant fields from the data set.

        Returns
        -------
        None.

        """
        self._titanic_data = self._titanic_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], axis = 1)
      
#%%        
    def deleteNaN(self):
        """
        Convert to numerical values.

        Returns
        -------
        None.

        """
        self._titanic_data = self._titanic_data.dropna(how='any')
        print(self._titanic_data.shape)
        
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
            data=self._titanic_data, 
            kind="count"
        )
        plt.close()

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
        x = self._titanic_data.drop(columns = column)
        y = self._titanic_data[column]
        
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size = 0.2, random_state = 0
        )
        
        print("x_train: ".ljust(80, '-'), "\n")
        print(self._x_train)
        print("\n" + "x_test: ".ljust(80, '-'), "\n")
        print(self._x_test)
        print("\n" + "y_train: ".ljust(80, '-'), "\n")
        print(self._y_train)
        print("\n" + "y_test: ".ljust(80, '-'), "\n")
        print(self._y_test)
        
#%%
    def regressionModel(self):
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
    def decisionTreeModel(self):
        """
        Apply decision tree model.

        Returns
        -------
        None.

        """
        self._dTree = DecisionTreeClassifier()
        self._dTree.fit(self._x_train, self._y_train)
        
#%%        
    def svcModel(self):
        """
        Apply SVC model.

        Returns
        -------
        None.

        """
        self._model = SVC()
        self._model.fit(self._x_train, self._y_train)
        
#%%        
    def neighborsModel(self):
        """
        Apply neighbors model.

        Returns
        -------
        None.

        """
        self._model = KNeighborsClassifier(
            n_neighbors=4
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
        if model == 'Logistic':
            self._y_pred = self._model.predict(self._x_test)
            
        elif model == 'Tree':  
            self._y_pred = self._dTree.predict(self._x_test)
            
        elif model == 'SVC':  
            self._y_pred = self._model.predict(self._x_test)
            
        elif model == 'Neighbors':
            self._y_pred = self._model.predict(self._x_test)
        
        print(self._y_pred)

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
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        class_names = ['Not survived', 'Survived']
        dataframe = pd.DataFrame(self._cm, index = class_names, columns = class_names)
        sns.heatmap(dataframe, annot = True, fmt='', cmap='Blues')
        
        #plt.close()
        
#%%        
    def showScore(self):
        """
        Show the score.

        Returns
        -------
        None.

        """
        if model == 'Logistic':
            print(
                "Model score: ", 
                self._model.score(
                    self._x_train, self._y_train
                ) * 100
            )
            
        elif model == 'Tree':    
            print(
                "Model score: ", 
                self._dTree.score(
                    self._x_train, self._y_train
                ) * 100
            )  
            
        elif model == 'SVC':
            print(
                "Model score: ", 
                self._model.score(
                    self._x_train, self._y_train
                ) * 100
            )
            
        elif model == 'Neighbors':
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
    titanic.loadfile()
    print("   ↑ 100% \n")
    
    titanic.showGeneralInformation()
    
    titanic.showDF()
    
    titanic.checkRelevantInfo()
    
    titanic.showDescription()
    
    print("1.2 Check non-numerical values".ljust(80, '-'), "\n")
    
    #Check the different values that exist.
    print("1.2.1 Check the different values that exist".ljust(80, '-'), "\n")
    
    print("Gender:")
    print(titanic.uniqueValues("Sex"))
    print("\n")
    
    print("Embarked:")
    print(titanic.uniqueValues("Embarked"))
    print("\n")
    
    #Assign a different value to each value.
    print("1.2.2 Assign a different value to each value:".ljust(80, '-'), "\n")
    titanic.replaceTags('Sex')
    titanic.replaceTags('Embarked')
    
    titanic.showDF()
    
    #1.3 Converting an object to a numerical data (diagnosis).
    print("1.3 Converting an object to a numerical ".ljust(80, '-'), "\n")
    titanic.toNumerical('Sex')
    titanic.toNumerical('Embarked')
    
    titanic.showGeneralInformation()
    
    #1.4 Round age.
    print("1.4 Round age ".ljust(80, '-'), "\n")
    print(titanic.checkNaN())
    print("\n")
    
    #1.5 Choose the most relevant fields from the data set.
    #Delete the columns.
    print("1.5 Choose the most relevant fields from the data set ".ljust(80, '-'), "\n")
    titanic.deleteColumns()
    print("\n")
    
    titanic.checkRelevantInfo()
    
    #1.6 Check for NaN data.
    print("1.6 Check for NaN data ".ljust(80, '-'), "\n")
    print(titanic.checkNaN())
    print("\n")
    
    #1.7 Delete NaN values.
    print("1.7 Delete NaN values ".ljust(80, '-'), "\n")
    titanic.deleteNaN()
    print("\n")
    
    titanic.showDF()
    
#%%    
def trainModel(model):
    """
    Train the model.

    Returns
    -------
    None.
    
    """
    print("2. Train the model ".ljust(80, '-'), "\n")
    
    #2.1 Graphics.
    print("2.1 Graphics: ".ljust(80, '-'), "\n")
    
    titanic.plotUsingSNS('Survived')
    
    #2.2 Split data.
    print("2.2 Split data: ".ljust(80, '-'), "\n")
    titanic.splitData('Survived')
    print("\n")
    
    if model == 'Logistic':
        #2.3 Apply logistic regression model.
        print("2.3 Apply logistic regression model: ".ljust(80, '-'), "\n")
        titanic.regressionModel()
        
    elif model == 'Tree': 
        #2.3 Apply desicion three classifier model.
        print("2.3 Apply desicion three classifier model: ".ljust(80, '-'), "\n")
        titanic.decisionTreeModel()
        
    elif model == 'SVC': 
        #2.3 Apply SVC model.
        print("2.3 Apply SVC model: ".ljust(80, '-'), "\n")
        titanic.svcModel()
    
    elif model == 'Neighbors':
        #2.3 Apply neighbors classifier model.
        print("2.3 Apply neighbors classifier model: ".ljust(80, '-'), "\n")
        titanic.neighborsModel()
        
#%%
def testModel(model):
    """
    Test model.

    Parameters
    ----------
    model : string
        Model type.

    Returns
    -------
    None.

    """
    print("3. Test model ".ljust(80, '-'), "\n")
    
    #3.1 Predictions.
    print("3.1 Predictions: ".ljust(80, '-'), "\n")
    titanic.prediction()
    print("\n")
    
    #3.2 Confusion matrix.
    print("3.2 Confusion matrix: ".ljust(80, '-'), "\n")
    titanic.confusionMatrix()
    print(titanic._cm)
    print("\n")
    
    #3.3 Plot confusion matrix.
    print("3.3 Plot confusion matrix: ".ljust(80, '-'), "\n")
    titanic.plotMatrixPrettier()
    
    #3.4 Show model score.
    print("3.4 Show model score: ".ljust(80, '-'), "\n")
    titanic.showScore()
    print("\n")
    
    #3.5 Show model acurracy.
    print("3.5 Show model acurracy: ".ljust(80, '-'), "\n")
    titanic.showAccuracy()
    print("\n")

#%%    
if __name__ == "__main__":
    global titanic
    titanic = Titanic(".", "train.csv")
    #model = 'Logistic'
    #model = 'Tree'
    #model = 'SVC'
    model = 'Neighbors'
    
    prepareData()
    trainModel(model)
    testModel(model)