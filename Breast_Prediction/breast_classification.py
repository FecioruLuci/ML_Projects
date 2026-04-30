import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np

tf.random.set_seed(3)
dataset_path = r"W:\vscode\SQL\MachineLearningProject\Breast_Cancer_Classification\data.csv"

df = pd.read_csv(dataset_path)
#print(df.head(10))
# in case we use the dataset for postgresql 
df = df.rename(columns={"concave points_mean":"concave_points_mean", "concave points_se":"concave_points_se", "concave points_worst":"concave_points_worst"})
#print(df.info())
#print(df.head(6))
df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
df = pd.get_dummies(df,columns=["diagnosis"],drop_first=True,dtype=int)
df = df.rename(columns={"diagnosis_M":"target"})
#print(df["target"].head(10))
#print(df.isnull().sum())
#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df["target"].value_counts())
#print(df.groupby("target").mean())

X = df.drop(columns="target")
Y = df["target"]
#print(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=41)
print(X.shape, x_test.shape, x_train.shape)
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.fit_transform(x_test)

#print(x_train_std.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train_std, y_train, validation_split = 0.1, epochs = 10)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Accuracy difference")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(['train', 'validation data'],loc = 'lower right')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title("Loss difference")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(['los', 'loss test data'],loc = 'upper right')
# plt.show()

loss, accuracy = model.evaluate(x_test_std, y_test)

#print(f"Accuracy = {accuracy} and loss = {loss}")

y_pred = model.predict(x_test_std)
#print(y_pred.shape)
y_pred_list = []
for i in y_pred:
    y_pred_label = np.argmax(i)
    if y_pred_label == 0:
        y_pred_list.append(0)
    else:
        y_pred_list.append(1)

#print(y_pred_list)

input_data = ()
input_data_as_nparray = np.asarray(input_data)