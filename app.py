from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Emoji dictionary for prediction output
emoji_dictionary = {
    "0": "\u2764\uFE0F",  # ❤️
    "1": ":baseball:",  # ⚾
    "2": ":beaming_face_with_smiling_eyes:",  # 😁
    "3": ":downcast_face_with_sweat:",  # 😓
    "4": ":fork_and_knife:"  # 🍴
}

# Load GloVe embeddings
embeddings = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coeffs

# Preprocessing function to convert text to embedding matrix
def get_output_embeddings(X, max_len=10):
    embedding_matrix_output = np.zeros((len(X), max_len, 50), dtype=np.float32)
    for ix in range(len(X)):
        words = X[ix].split()
        for jx, word in enumerate(words[:max_len]):  # Only process up to max_len words
            if word.lower() in embeddings:
                embedding_matrix_output[ix][jx] = embeddings[word.lower()]
    return embedding_matrix_output

# Function to process and predict using the TFLite model
def predict_text(input_text):
    # Convert the input text to embeddings
    processed_input = get_output_embeddings([input_text])  # Wrap text in a list as input
    processed_input = np.array(processed_input, dtype=np.float32)  # Ensure input is float32

    # Set model input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], processed_input)

    # Run the model
    interpreter.invoke()

    # Get model output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class (index of the highest probability)
    predicted_class = np.argmax(output_data, axis=1)[0]
    predicted_emoji = emoji_dictionary.get(str(predicted_class), 'Unknown')
    
    return predicted_emoji

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the POST request
    data = request.json
    text = data['text']

    # Predict the emoji for the given text
    prediction = predict_text(text)

    # Return the result as JSON
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
