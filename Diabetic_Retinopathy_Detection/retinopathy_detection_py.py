import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


df_labels = pd.read_csv(r"W:\vscode\SQL\MachineLearningProject\Diabetic_Retinopathy_Detection\trainLabels_cropped.csv",index_col=0)

#print(df_labels.head(7))

df_labels_cleared = df_labels[['image', 'level']]
#print(df_labels_cleared.shape) -- am realizat faptul ca acest index este invizibil pentru model.

#print(df_labels_cleared)

path = r"W:\vscode\SQL\MachineLearningProject\Diabetic_Retinopathy_Detection\train\resized_train_cropped"
files = glob.glob(os.path.join(path, "**/*.jpeg"),recursive=True)
file_count = len(files)
#print(file_count)  -- avem 35108 imagini pentru antrenare

#print(df_labels_cleared.groupby("level").size())

# 0 - No DR -25802-
# 1 - Mild  -2438-
# 2 - Moderate  -5288-
# 3 - Severe    -872-
# 4 - Proliferative DR  -708-

# plt.figure(figsize=(15,20))

# for i in range(5):
#     referinta = df_labels_cleared[df_labels_cleared["level"] == i].iloc[0]
#     img_nume = str(referinta["image"])
#     if not img_nume.lower().endswith(".jpeg"):
#         img_nume += ".jpeg"
#     img_path = os.path.join(path, img_nume)

#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #img_blur = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

#     plt.subplot(1,5,i+1)
#     plt.imshow(img)
#     plt.title(f"Level {i}")
#     plt.axis("off")

# plt.show()
# print(referinta)
file_paths = []
for image_name in df_labels_cleared["image"]:
    if not image_name.endswith(".jpeg"):
        image_name = image_name + ".jpeg"
    file_paths.append(os.path.join(path, image_name))

df_labels_cleared["file_paths"] = file_paths
#print(df_labels_cleared.head(6))
#print(df_labels_cleared["file_paths"])

# print(len(x_train),len(y_train), len(x_test), len(y_test))

#print(class_weight_dict)

def procesare_imagine(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    return img

test_img = procesare_imagine(r"W:\vscode\SQL\MachineLearningProject\Diabetic_Retinopathy_Detection\train\resized_train_cropped\22_right.jpeg")
#print(test_img.shape, test_img, test_img.min(), test_img.max())    #224/224 3-RGB iar pixelii sunt de la 0 care este pixel negru pana la 0.964 care este aproape de pixel 255 - alb

df_balansat_700 = pd.DataFrame()

for level in range(0,5):
    df_level = df_labels_cleared[df_labels_cleared["level"] == level]
    numere = min(len(df_level), 1500)
    df_sample = df_level.sample(n=numere, random_state=41)
    df_balansat_700 = pd.concat([df_balansat_700, df_sample])

df_balansat_700 = df_balansat_700.reset_index(drop=True)
#print(df_balansat_700.groupby("level").size()) 
#print(df_balansat_700.head(7))
# 1500 - 0; 1500 - 1; 1500 - 2; 872 - 3; 708 - 4


x_train, x_test, y_train, y_test = train_test_split(df_balansat_700["file_paths"], df_balansat_700["level"], test_size=0.2, stratify=df_balansat_700["level"], random_state=41)

#print(f"Avem {len(x_train)} imagini de train si {len(x_test)} imagini de test")
clase = np.asarray([0,1,2,3,4])
weight = compute_class_weight(class_weight="balanced", classes=clase, y=y_train)
class_weight_dict = dict(zip(clase,weight))

x_train_list = []

for filepath_train in x_train:
    img = procesare_imagine(filepath_train)
    x_train_list.append(img)

x_train_list = np.array(x_train_list)

x_test_list = []

for filepath_test in x_test:
    img = procesare_imagine(filepath_test)
    x_test_list.append(img)

x_test_list = np.array(x_test_list)

#print(x_test_list.shape, x_train_list.shape)
# avem 700 imagini de test (0.2 * train ) si 2800 pentru train exact cum vrem noi total(3500) - 02 * train

y_test = np.array(y_test)
y_train = np.array(y_train)
#print(test_img.min(), test_img.max())
#print(y_test.shape, y_train.shape)
# am facut si datele ce nu le stie adica labels ca array pentru a putea face comparatia
#print(x_train_list.shape)
base_model = EfficientNetB3(weights = "imagenet", include_top = False, input_shape = (224,224,3))

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training = False)
x = GlobalAveragePooling2D() (x)
x = Dropout(0.3) (x)
x = Dense(128, activation="relu") (x)
x = Dropout(0.3) (x)
output = Dense(5, activation="softmax") (x)

model = Model(inputs = inputs, outputs = output)
model.summary()
# am declarat x variabila de output a modelului, dupa l am facut intr un vector 2D, dupa l am facut sa nu invete 30% 
# (pentru stabilitate sa nu avem overfitting) dupa am conectat toate straturile, dupa am setat 5 outputuri (de la 5 levels 0,1,2,3,4) 
# pentru clasa


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train_list, y_train, epochs=15, batch_size=32, validation_data=(x_test_list, y_test), class_weight=class_weight_dict)
