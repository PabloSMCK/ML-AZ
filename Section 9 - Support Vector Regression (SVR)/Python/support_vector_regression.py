"""PROBLEM: WE HAVE TO HIRE A PERSON TAHT WANTS 160000 FOR SALARY, WE HAVE TO CHECK IF IN HIS PREVIOUS JOB 
HE HAD THAT SAME AMOUNT THAT HE IS ASKING, WE ARE GOING TO COMPARE THE RESULTS WITH SUPORT VECTOR MACHINE,
 HE CLAIMS TO HAVE 6.5 YEARS OF EXPERIENCE 
 

SVR USA FEATURE SCALLING 
 
"""


#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1].values #INDEPENDET VARIABLE/S
y = dataset.iloc[:,-1].values #DEPENDENT VARIABLE/S

#TRANSFORM IN A 2D ARRAY
X = np.reshape(X,(len(X),1))
y = np.reshape(y,(len(y),1))


#SPLITTING THE DF TO TRAIN AND TEST
"""EN ESTE CASO NO HAY PORQUE HAY POCOS DATOS Y HA YQUE MANTENERLOS"""

#FEATURE SCALLING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X  = sc_X.fit_transform(X)
sc_y = StandardScaler()               
y  = sc_y.fit_transform(y)

#TRAIN THE SVR MODEL
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#PREDICT A NEW RESULT
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))

#VISUALIZING THE VSR RESULTS
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#VISUALIZING THE VSR RESULTS WITH PRETIER GRAPHICS

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
