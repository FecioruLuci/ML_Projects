import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"W:\vscode\SQL\MachineLearningProject\Solar_Rock_Mine\Copy of sonar data.csv", header=None)


#print(df.head(6))
#print(df.shape)
#print(df.describe())
#print(df[60].value_counts())

x = df.drop(columns=60)
y = df[60]

#print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=1)
#print(x.shape,x_train.shape, x_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_train)
training_data_acc = accuracy_score(prediction, y_train)
#print(training_data_acc)
prediction_test = model.predict(x_test)
training_data_acc_test = accuracy_score(prediction_test, y_test)
#print(training_data_acc_test)

input_data = (0.0715,0.0849,0.0587,0.0218,0.0862,0.1801,0.1916,0.1896,0.2960,0.4186,0.4867,0.5249,0.5959,0.6855,0.8573,0.9718,0.8693,0.8711,0.8954,0.9922,0.8980,0.8158,0.8373,0.7541,0.5893,0.5488,0.5643,0.5406,0.4783,0.4439,0.3698,0.2574,0.1478,0.1743,0.1229,0.1588,0.1803,0.1436,0.1667,0.2630,0.2234,0.1239,0.0869,0.2092,0.1499,0.0676,0.0899,0.0927,0.0658,0.0086,0.0216,0.0153,0.0121,0.0096,0.0196,0.0042,0.0066,0.0099,0.0083,0.0124)

input_date_as_numpy_arr = np.asarray(input_data)
input_data_reshaped = input_date_as_numpy_arr.reshape(1, -1)

prediction2 = model.predict(input_data_reshaped)

print(prediction2)