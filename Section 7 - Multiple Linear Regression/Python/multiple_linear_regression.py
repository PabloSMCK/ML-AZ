#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT THE DATASET
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values #INDEPENDET VARIABLE/S
y = dataset.iloc[:,-1].values #DEPENDENT VARIABLE/S


#TAKING CARE OF THE MISSING DATA
"""from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform( X[:, 1:3])"""

#ENCODING CATEGORICAL (TRANSFORM COLUMN TEXT VALUES TO NUMBERS)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
"""labelencoder_y = LabelEncoder() #ALTERA LA VARIABLE y
y = labelencoder_y.fit_transform(y)#ALTERA LA VARIABLE y"""


#SPLITTING THE DF TO TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#FEATURE SCALLING
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train  = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#PREDICT THE TEST SET RESULTS
y_pred = regressor.predict(X_test) #VECTOR WITH PREDICTED VALUES
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)) #CONCATENAR LOS 2 VECTORES, RESHAPE = TRANSFORMAR LOS VECTORES A LO LARGO EN COLUMNAS, LOS 1 SON LOS AXIS
