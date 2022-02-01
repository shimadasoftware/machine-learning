#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 08:55:55 2021

@author: juana
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings

class Degree():
    
#%%
    def __init__(self):
        """
        Default Constructor.

        Returns
        -------
        None.

        """
        self._celcius = np.array([-40, -20, -10, 0, 10, 15, 20, 25, 30, 35], dtype = float)
        self._fahrenheit = np.array([-40, -4, 14, 32, 50, 59, 68, 77, 86, 95], dtype = float)
        
#%%        
    def createShape(self, units, shape):
        """
        Create shape.

        Returns
        -------
        None.

        """
        self._shape = tf.keras.layers.Dense(units = units, input_shape = [shape])

#%%        
    def createModel(self):
        """
        Create model.

        Returns
        -------
        None.

        """
        self._model = tf.keras.Sequential([self._shape])

#%%        
    def compileModel(self, learningMug, loss):
        """
        Compile model.

        Returns
        -------
        None.

        """
        self._model.compile(
            optimizer = tf.keras.optimizers.Adam(learningMug),
            loss = loss
        )
        
#%%        
    def startModel(self, epochs):
        """
        Start trainig model.

        Returns
        -------
        None.

        """
        
        self._data_train = self._model.fit(
            self._celcius, 
            self._fahrenheit, 
            epochs = epochs, 
            verbose = False
        ) 
        
#%%        
    def plotModel(self, x, y, loss):
        """
        Plot model.

        Returns
        -------
        None.

        """
        plt.xlabel(x)
        plt.ylabel(y)
        plt.plot(self._data_train.history[loss])

#%%        
    def predictions(self, loops):
        """
        Compile model.

        Returns
        -------
        None.

        """
        y_pred = self._model.predict([loops])
        print("The result approximately is: ", str(y_pred), "°F")

#%%        
    def showValues(self):
        """
        Show values.

        Returns
        -------
        None.

        """
        print("Weight and bias: ", self._shape.get_weights())

#%%        
    def createHiddenShapes(self):
        """
        Create hidden shapes.

        Returns
        -------
        None.

        """
        hide1 = tf.keras.layers.Dense(units = 4, input_shape = [1])
        hide2 = tf.keras.layers.Dense(units = 4)
        output = tf.keras.layers.Dense(units = 1)
        
        self._model = tf.keras.Sequential([hide1, hide2, output])

#%%
def prepareData():
    """
    Prepare the data.

    Returns
    -------
    None.

    """
    print("1. Prepare data: ".ljust(90, '-'), "\n")
    
    print("1.1 Create shape: ".ljust(90, '-'), "\n")
    degree.createShape(1, 1)
    print("\n")
    
    print("1.2 Create model: ".ljust(90, '-'), "\n")
    degree.createModel()
    print("\n")
    
    print("1.3 Compile model: ".ljust(90, '-'), "\n")
    degree.compileModel(0.1, 'mean_squared_error')
    print("\n")
    
#%%    
def trainModel():
    """
    Train the model.

    Returns
    -------
    None.
    
    """
    print("2. Train the model ".ljust(90, '-'), "\n")
    
    print("2.1 Start trainig model: ".ljust(90, '-'), "\n")
    degree.startModel(500)
    print("\n")
    
    print("2.2 Start trainig model: ".ljust(90, '-'), "\n")
    degree.plotModel("Epochs", "Starting magnitude", "loss")
    print("\n")   

    
#%%
def testModel():
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
    print("3. Test model ".ljust(90, '-'), "\n")
    
    print("3.1 Predictions: ".ljust(90, '-'), "\n")
    degree.predictions(100)
    print("\n")  
    
    print("3.2 Show values: ".ljust(90, '-'), "\n")
    degree.showValues()
    print("\n")  
    
    print("3.3 Create hidden shapes: ".ljust(90, '-'), "\n")
    degree.createHiddenShapes()
    print("\n") 
    
    print("1.3 Compile model: ".ljust(90, '-'), "\n")
    degree.compileModel(0.1, 'mean_squared_error')
    print("\n")
    
    print("2. Train the model ".ljust(90, '-'), "\n")
    
    print("2.1 Start trainig model: ".ljust(90, '-'), "\n")
    degree.startModel(500)
    print("\n")
    
    print("2.2 Start trainig model: ".ljust(90, '-'), "\n")
    degree.plotModel("Epochs", "Starting magnitude", "loss")
    print("\n") 
    
    print("3.1 Predictions: ".ljust(90, '-'), "\n")
    degree.predictions(100)
    print("\n")  

#%%    
if __name__ == "__main__":
    global degree
    degree = Degree()
    
    prepareData()
    trainModel()
    testModel()