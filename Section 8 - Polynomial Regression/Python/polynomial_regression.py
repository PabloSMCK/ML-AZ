"""PROBLEM: WE HAVE TO HIRE A PERSON TAHT WANTS 160000 FOR SALARY, WE HAVE TO CHECK IF IN HIS PREVIOUS JOB 
HE HAD THAT SAME AMOUNT THAT HE IS ASKING, WE ARE GOING TO COMPARE THE RESULTS BETWEEN A LINEAR MODEL AND A
POLYNOMIAL REGRESSION MODE, HE CLAIMS TO HAVE 6.5 YEARS OF EXPERIENCE 
"""

#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1].values #INDEPENDET VARIABLE/S
y = dataset.iloc[:,2].values #DEPENDENT VARIABLE/S

X = X.reshape(len(X),1)
y = y.reshape(len(y),1)
#TAKING CARE OF THE MISSING DATA
"""from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform( X[:, 1:3])"""

#ENCODING CATEGORICAL (TRANSFORM COLUMN TEXT VALUES TO NUMBERS)
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""


#SPLITTING THE DF TO TRAIN AND TEST
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


#FEATURE SCALLING
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train  = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)"""

#CREATE THE LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#CREATE THE POLYNOMIAL REGRESSION MODEL
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #DEGREE ES EL ELEVADO QEU TIENE LA X EN LA EC POLYNOMIAL
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#PREDICT THE TEST SET RESULTS
y_pred = lin_reg.predict(X) #VECTOR WITH PREDICTED VALUES

#VISUALIZING THE TRAINING SET RESULTS LINEAR
plt.scatter(X, y, color = 'red')
plt.plot(X,y_pred, color = 'blue')
plt.show()

#VISUALIZING THE TRAINING SET RESULTS POLYNOMIAL
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


#PREDICTING A VALUE WITH LINEAR REGRESSION EJ: 6.5
print(lin_reg.predict(([[6.5]])))

#PREDICTING A VALUE WITH POLYNOMIAL REGRESSION EJ: 6.5
print(lin_reg_2.predict(poly_reg.fit_transform(([[6.5]]))))