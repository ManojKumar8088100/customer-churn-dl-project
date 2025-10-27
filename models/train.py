import pandas as pandas
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import standardScaler
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_scv('data/customer_churn.csv')

x = df.drop(['CustomerId','Surname','Exited'],axis=1)

y = df['Exited']

