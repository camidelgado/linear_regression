from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

url = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
df = pd.read_csv(url, sep=',')

sex_dic = {'male': 1, 'female': 0}
df['sex'] = df['sex'].map(sex_dic)

smoker_dic = {'yes': 1, 'no': 0}
df['smoker'] = df['smoker'].map(smoker_dic)

region_dic = {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}
df['region'] = df['region'].map(region_dic)

df['charges_log'] = np.log10(df['charges']+10**(-6))
X=df.drop(columns=['charges','charges_log'])
y=df['charges']
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=412)
mi_modelo=LinearRegression()
mi_modelo.fit(X_train,y_train)
y_pred=mi_modelo.predict(X_test)
y_train_pred=mi_modelo.predict(X_train)

