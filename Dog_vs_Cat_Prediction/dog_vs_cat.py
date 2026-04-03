import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

path = r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\train"

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
file_names = os.listdir(r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\train")

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

#os.mkdir(r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\img_resized")

original_folder = r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\train"
resized_folder = r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Dog_vs_Cat_Prediction\dataset\img_resized"


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

    print(img_path)

    img = Image.open(img_path)
    img = img.resize((224,224))
    img = img.convert("RGB")
    new_img_path = resized_folder + "\\" + file_name
    img.save(new_img_path)

    if file_name.startswith("cat"):
        cat_counter += 1
    else:
        dog_counter += 1
    
print("Total cats resized: ",cat_counter)
print("Total dogs resized: ",dog_counter)


