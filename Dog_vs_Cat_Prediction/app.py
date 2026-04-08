from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model/model_cat_dog.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.reshape(img, [1, 224, 224, 3]).astype(np.float32)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    label_index = np.argmax(prediction)
    label = "dog" if label_index == 1 else "cat"
    confidence = float(prediction[0][label_index])
    
    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)