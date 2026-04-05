import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.applications import MobileNetV2
from keras import layers, models

path = r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\train\train"

files = glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
file_count = len(files)

#print("Total number of images: ",file_count)

# all_files = []
# categories = ['cats', 'dogs']

# for category in categories:
#     folder_path = os.path.join(path, category)
#     files = os.listdir(folder_path)
#     all_files.extend(files)
#     print(f"Am găsit {len(files)} imagini în folderul '{category}'")

# img_cat = mpimg.imread(r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\train\cats\cat.2.jpg")
# img_dog = mpimg.imread(r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\train\dogs\dog.15.jpg")
# imgplot = plt.imshow(img_dog)
# plt.show()
file_names = os.listdir(r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\train\train")

#print(file_names)

for i in range(5):
    name = file_names[i]
    #(name[0:3])

dog_count = 0
cat_count = 0

for img_file in file_names:
    name = img_file[0:3]
    if name == 'dog':
        dog_count += 1
    else:
        cat_count += 1

#print("Number of dog images = ",dog_count)
#print("Number of cat images = ",cat_count)

#os.mkdir(r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\img_resized")

original_folder = r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\train\train"
resized_folder = r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\img_resized"


file_names = os.listdir(original_folder)
cats = []
dogs = []
for word in file_names:
    if word.startswith('cat'):
        cats.append(word)
    else:
        dogs.append(word) 
selected = cats[:1000] + dogs[:1000]


cat_counter = 0
dog_counter = 0

for file_name in selected:
    img_path = original_folder + "\\" + file_name

    #print(img_path)

    img = Image.open(img_path)
    img = img.resize((224,224))
    img = img.convert("RGB")
    new_img_path = resized_folder + "\\" + file_name
    img.save(new_img_path)

    if file_name.startswith("cat"):
        cat_counter += 1
    else:
        dog_counter += 1
    
# print("Total cats resized: ",cat_counter)
# print("Total dogs resized: ",dog_counter)

# img = mpimg.imread(r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\img_resized\dog.10228.jpg")
# imgplot = plt.imshow(img)
# plt.show()

# cat -> 0; dog -> 1
filenames = os.listdir(resized_folder)
labels = []
#print(filenames)

for i in range(2000):
    file_name = filenames[i]
    label = file_name[0:3]

    if label == 'dog':
        labels.append(1)
    else:
        labels.append(0)
count_cat = 0
count_dog = 0
for label in labels:
    if label == 1:
        count_dog += 1
    else:
        count_cat += 1

# print("Cats: ",count_cat)
# print("Dogs: ",count_dog)

files = []
image_extension = ['*.png', '*.jpg']
for e in image_extension:
    pattern = os.path.join(resized_folder,e)
    found_files = glob.glob(pattern)
    files.extend(found_files)
data = []

for file in files:
    img = cv2.imread(file)
    if img is not None: 
        data.append(np.asarray(img))
    else:
        print(f"Nu s-a putut face transformarea fisierului: {file}")
dog_cat_images = np.asarray(data)

#print(dog_cat_images)

print(dog_cat_images.shape)


X = dog_cat_images
Y = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

print(X.shape, x_train.shape, x_test.shape)

x_train_scaled = x_train / 255
x_test_scaled = x_test / 255


base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')


base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_scaled, y_train, epochs=5)

score, acc = model.evaluate(x_test_scaled, y_test)

print(f"Test loss = {score} and accuracy = {acc}")

input_img_path = input("Path of the image: ")
input_image = cv2.imread(input_img_path)

cv2.imshow('Imagine' ,input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

input_image_resize = cv2.resize(input_image,(224,224))
input_image_rgb = cv2.cvtColor(input_image_resize, cv2.COLOR_BGR2RGB)
input_image_resize_scaled = input_image_rgb / 255
image_reshape = np.reshape(input_image_resize_scaled, [1,224,224,3])
input_prediction = model.predict(image_reshape)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 0:
    print("The image is a cat")
else:
    print("Image is a dog")


print(input_prediction)


