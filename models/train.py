import pandas as pandas
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import standardScaler,LabelEncoder
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_scv('data/customer_churn.csv')

x = df.drop(['CustomerId','Surname','Exited'],axis=1)

y = df['Exited']

x['Geography']=LabelEncoder().fit_transform(x['Geography'])
x['Gender'] = LabelEncoder().fit_transform(x['Gender'])
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

scaler = standardScaler()
x_train = scaler.fit_transform(x_train)
