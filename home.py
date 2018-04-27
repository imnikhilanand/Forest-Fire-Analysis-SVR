# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:38:01 2018

@author: Sonal
"""

#importing libraries

import numpy as np
import pandas as pd 

dataframe = pd.read_csv("forestfires.csv")

x = dataframe.iloc[:,[0,1,4,5,6,7,8,9,10,11]].values
y = dataframe.iloc[:,[12]].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
x_train = scX.fit_transform(x_train)
x_test = scX.fit_transform(x_test)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


