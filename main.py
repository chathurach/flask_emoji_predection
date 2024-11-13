import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf

# Load and preprocess data
train_data = pd.read_csv('train_emoji.csv', header=0)  # Skip header row
test_data = pd.read_csv('test_emoji.csv', header=0)    # Skip header row

# Emoji dictionary
emoji_dictionary = {
    "0": "\u2764\uFE0F",  # ❤️
    "1": ":baseball:",  # ⚾
    "2": ":beaming_face_with_smiling_eyes:",  # 😁
    "3": ":downcast_face_with_sweat:",  # 😓
    "4": ":fork_and_knife:"  # 🍴
}

# Print sample data
for i in range(min(10, len(train_data))):
    print(train_data.iloc[i]['Text'], "-----", emoji_dictionary.get(str(train_data.iloc[i]['Labels']), "Unknown"))

# Prepare training and test data
XT = train_data['Text'].values
Xt = test_data['Text'].values

def convert_labels(labels):
    labels = pd.to_numeric(labels, errors='coerce')  # Convert to numeric, set errors to NaN
    return labels.fillna(0).astype(int)  # Replace NaNs with 0 and convert to int

YT = to_categorical(convert_labels(train_data['Labels']))
Yt = to_categorical(convert_labels(test_data['Labels']))

# Load GloVe embeddings
embeddings = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coeffs

def get_output_embeddings(X, max_len=10):
    embedding_matrix_output = np.zeros((len(X), max_len, 50))
    for ix in range(len(X)):
        words = X[ix].split()
        for jx, word in enumerate(words[:max_len]):  # Only process up to max_len words
            if word.lower() in embeddings:
                embedding_matrix_output[ix][jx] = embeddings[word.lower()]
    return embedding_matrix_output

# Convert text to embeddings
emb_XT = get_output_embeddings(XT)
emb_Xt = get_output_embeddings(Xt)

# Define and compile the model
model = Sequential()
model.add(LSTM(128, input_shape=(10, 50), return_sequences=True))  # Increased units
model.add(Dropout(0.5))  # Increased dropout rate
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Moved activation to Dense layer

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(emb_XT, YT, batch_size=32, epochs=40, shuffle=True, validation_split=0.1)

# Evaluate the model
evaluation = model.evaluate(emb_Xt, Yt)
print(f"Evaluation result: {evaluation}")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

try:
    tflite_model = converter.convert()
    with open('converted_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model conversion to TensorFlow Lite format successful.")
except Exception as e:
    print(f"Error during TensorFlow Lite conversion: {e}")

# Check if model conversion was successful
try:
    interpreter = tf.lite.Interpreter(model_path='D:\converted_model.tflite')
    interpreter.allocate_tensors()
    print("TensorFlow Lite model loaded and tensors allocated successfully.")
except Exception as e:
    print(f"Error loading TensorFlow Lite model: {e}")

# Predict and print results
pred_probs = model.predict(emb_Xt)
pred = np.argmax(pred_probs, axis=1)
for i in range(min(30, len(Xt))):  # Adjusted to handle fewer test samples
    print(f"Text: {Xt[i]}")
    print(f"Prediction: {emoji_dictionary.get(str(pred[i]), 'Unknown')}")
