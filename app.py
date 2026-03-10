# ============================================
# STEP 3: Flask Backend (Web Server)
# ============================================
# This runs the web app and predicts sentiment
# when a user types a movie review.
# ============================================

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pickle
import keras
from flask import Flask, render_template, request, jsonify

# ---------------------------
# 1. Create the Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# 2. Load the trained model and tokenizer
# ---------------------------
print("📂 Loading model and tokenizer...")
model = keras.saving.load_model("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    word_index = pickle.load(f)

with open("model/accuracy.txt", "r") as f:
    model_accuracy = f.read().strip()

with open("model/max_length.txt", "r") as f:
    max_length = int(f.read().strip())

cm = np.load("model/confusion_matrix.npy")

print("✅ Model loaded successfully!")

# ---------------------------
# 3. Helper functions
# ---------------------------

def text_to_sequence(text, word_index):
    """Convert text to number sequence"""
    return [word_index.get(w, 1) for w in text.lower().split()]

def pad_sequence(seq, max_len):
    """Make all sequences the same length"""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

def detect_mixed_sentiment(text):
    """
    Check if the review has contrast words like
    'but', 'however', 'although' — which usually
    mean the review has MIXED feelings.
    """
    contrast_words = [
        "but", "however", "although", "though",
        "yet", "despite", "nevertheless", "while",
        "whereas", "except", "still", "unfortunately"
    ]
    text_lower = text.lower()
    for word in contrast_words:
        if word in text_lower.split():
            return True
    return False

# ---------------------------
# 4. Web Routes
# ---------------------------

@app.route("/")
def home():
    """Show the main webpage"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receive a review and return the sentiment"""
    data = request.get_json()
    review_text = data.get("review", "")

    if not review_text.strip():
        return jsonify({"error": "Please enter a review!"})

    # Convert text to numbers
    sequence = text_to_sequence(review_text, word_index)
    padded = np.array([pad_sequence(sequence, max_length)]).astype("int32")

    # Get model prediction
    prediction = model.predict(padded, verbose=0)
    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100

    # Label mapping
    labels = {0: "Positive 😊", 1: "Negative 😠", 2: "Mixed 😐"}
    sentiment = labels[predicted_class]

    # Check for mixed sentiment using contrast word detection
    has_contrast = detect_mixed_sentiment(review_text)
    if has_contrast and predicted_class != 2:
        # If contrast words found but model didn't predict mixed,
        # check if mixed probability is reasonable
        mixed_prob = float(prediction[0][2]) * 100
        if mixed_prob > 15:
            sentiment = "Mixed 😐"
            confidence = mixed_prob

    # Build confusion matrix as a list for the frontend
    cm_list = cm.tolist()

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "accuracy": model_accuracy,
        "confusion_matrix": cm_list,
        "probabilities": {
            "positive": round(float(prediction[0][0]) * 100, 2),
            "negative": round(float(prediction[0][1]) * 100, 2),
            "mixed": round(float(prediction[0][2]) * 100, 2)
        }
    })

# ---------------------------
# 5. Run the app
# ---------------------------
if __name__ == "__main__":
    print("\n🚀 Starting web server...")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
