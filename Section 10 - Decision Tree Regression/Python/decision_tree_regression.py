"""PROBLEM: WE HAVE TO HIRE A PERSON TAHT WANTS 160000 FOR SALARY, WE HAVE TO CHECK IF IN HIS PREVIOUS JOB 
HE HAD THAT SAME AMOUNT THAT HE IS ASKING, WE ARE GOING TO COMPARE THE RESULTS WITH SUPORT VECTOR MACHINE,
 HE CLAIMS TO HAVE 6.5 YEARS OF EXPERIENCE 
 

DECISION TREE NO USA FEATURE SCALLING 
 
"""
#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1].values #INDEPENDET VARIABLE/S
y = dataset.loc[:,'Salary'].values #DEPENDENT VARIABLE/S

X = np.reshape(X,(len(X),1))
y = np.reshape(y,(len(y),1))

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

#VISUALIZING THE VSR RESULTS WITH PRETIER GRAPHICS

#VISUALIZING THE VSR RESULTS WITH PRETIER GRAPHICS

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()