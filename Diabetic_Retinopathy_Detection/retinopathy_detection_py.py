import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2

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

plt.figure(figsize=(15,20))

for i in range(5):
    referinta = df_labels_cleared[df_labels_cleared["level"] == i].iloc[0]
    img_nume = str(referinta["image"])
    if not img_nume.lower().endswith(".jpeg"):
        img_nume += ".jpeg"
    img_path = os.path.join(path, img_nume)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_blur = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title(f"Level {i}")
    plt.axis("off")

plt.show()