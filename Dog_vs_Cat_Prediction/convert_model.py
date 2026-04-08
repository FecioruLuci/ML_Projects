import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights(r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\model\model_weights.weights.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(r"W:\vscode\SQL\MachineLearningProject\Dog_vs_Cat_Prediction\model\model_cat_dog.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversie completata!")