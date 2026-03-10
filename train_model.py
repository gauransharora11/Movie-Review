# ============================================
# STEP 2: Train the Deep Learning Model
# ============================================
# This file reads the movie reviews from CSV,
# trains an LSTM model, and saves it.
# ============================================

# Set Keras to use JAX backend (works with Python 3.14)
import os
os.environ["KERAS_BACKEND"] = "jax"

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import keras
from keras import layers

# ---------------------------
# 1. Load the dataset
# ---------------------------
print("📂 Loading dataset...")
data = pd.read_csv("data/movie.csv")
print(f"✅ Loaded {len(data)} reviews")
print(data.head())

# ---------------------------
# 2. Convert sentiment labels to numbers
# ---------------------------
# positive = 0, negative = 1, mixed = 2
label_map = {"positive": 0, "negative": 1, "mixed": 2}
data["label"] = data["sentiment"].map(label_map)

reviews = data["review"].values
labels = data["label"].values

# ---------------------------
# 3. Build a simple tokenizer (word → number)
# ---------------------------
print("\n🔤 Tokenizing text...")

# Build vocabulary from all reviews
word_index = {"<PAD>": 0, "<OOV>": 1}
idx = 2
for review in reviews:
    for word in review.lower().split():
        if word not in word_index:
            word_index[word] = idx
            idx += 1

vocab_size = len(word_index)
print(f"✅ Vocabulary size: {vocab_size}")

# Convert text to number sequences
def text_to_sequence(text, word_index):
    return [word_index.get(w, 1) for w in text.lower().split()]

sequences = [text_to_sequence(r, word_index) for r in reviews]

# ---------------------------
# 4. Pad sequences (make all same length)
# ---------------------------
max_length = 20

def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

padded = np.array([pad_sequence(s, max_length) for s in sequences])
print(f"✅ Padded sequences shape: {padded.shape}")

# ---------------------------
# 5. Split into training and testing data
# ---------------------------
# One-hot encode labels (e.g., 0 -> [1,0,0])
labels_encoded = keras.utils.to_categorical(labels, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    padded, labels_encoded, test_size=0.2, random_state=42
)
print(f"\n📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# Convert to float32 for JAX
X_train = X_train.astype("int32")
X_test = X_test.astype("int32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# ---------------------------
# 6. Build the LSTM Model
# ---------------------------
print("\n🧠 Building LSTM model...")
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=64),
    layers.LSTM(64),
    layers.Dropout(0.5),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Build the model by passing sample data shape
model.build(input_shape=(None, max_length))
model.summary()

# ---------------------------
# 7. Train the model
# ---------------------------
print("\n🚀 Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# ---------------------------
# 8. Evaluate the model
# ---------------------------
print("\n📈 Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Get predictions for confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# Classification Report
label_names = ["positive", "negative", "mixed"]
print("\n📋 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_names, zero_division=0))

# ---------------------------
# 9. Save everything
# ---------------------------
print("\n💾 Saving model and tokenizer...")

# Save the trained model
model.save("model/sentiment_model.keras")
print("✅ Model saved to: model/sentiment_model.keras")

# Save the tokenizer (word_index dictionary)
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(word_index, f)
print("✅ Tokenizer saved to: model/tokenizer.pkl")

# Save max_length for the web app
with open("model/max_length.txt", "w") as f:
    f.write(str(max_length))

# Save accuracy for the web app
with open("model/accuracy.txt", "w") as f:
    f.write(f"{accuracy * 100:.2f}")
print("✅ Accuracy saved to: model/accuracy.txt")

# Save confusion matrix
np.save("model/confusion_matrix.npy", cm)
print("✅ Confusion matrix saved to: model/confusion_matrix.npy")

print("\n🎉 DONE! Model training complete!")
print(f"🎯 Final Accuracy: {accuracy * 100:.2f}%")
