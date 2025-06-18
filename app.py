from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define emotional aspects
emotional_aspects = {
    'motivation': ['motivat', 'inspir', 'drive', 'enthusias', 'passion', 'eager', 'excit'],
    'satisfaction': ['satisf', 'content', 'happy', 'enjoy', 'proud', 'fulfill', 'achiev'],
    'challenge': ['challeng', 'difficult', 'hard', 'struggl', 'frustrat', 'stress', 'overwhelm'],
    'learning': ['learn', 'knowledge', 'skill', 'understand', 'educat', 'growth', 'develop'],
    'social': ['team', 'collaborat', 'friend', 'interact', 'communit', 'connect', 'relation']
}

# Initialize global variables
tokenizer = None
aspect_encoders = None
aspect_models = None
max_sequence_length = 100

# Load models and tokenizer
def load_saved_models():
    models = {}
    
    # Path to saved models
    model_dir = 'saved_models'
    
    # Load tokenizer
    with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open(os.path.join(model_dir, 'aspect_encoders.pickle'), 'rb') as handle:
        aspect_encoders = pickle.load(handle)
    
    # Load models for each aspect
    for aspect in emotional_aspects.keys():
        model_path = os.path.join(model_dir, f'{aspect}_model.keras')
        if os.path.exists(model_path):
            models[aspect] = load_model(model_path)
    
    return tokenizer, aspect_encoders, models

# Text preprocessing functions
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(f'[{string.punctuation}]', ' ', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Sentiment prediction function
def predict_aspect_sentiment(text, aspect):
    if tokenizer is None or aspect not in aspect_models or aspect not in aspect_encoders:
        return 'Model not loaded'
    
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    
    # Check if aspect is mentioned in the text
    aspect_present = any(re.search(r'\b' + keyword + r'[a-z]*\b', processed) 
                         for keyword in emotional_aspects[aspect])
    if not aspect_present:
        return 'not_mentioned'
    
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=max_sequence_length)
    
    prediction = aspect_models[aspect].predict(padded)[0]
    
    # Handle binary or multiclass classification
    encoder = aspect_encoders[aspect]
    if len(encoder.classes_) == 2:
        idx = 1 if prediction > 0.5 else 0
    else:
        idx = np.argmax(prediction)
    
    return encoder.classes_[idx]

def initialize_models():
    global tokenizer, aspect_encoders, aspect_models
    tokenizer, aspect_encoders, aspect_models = load_saved_models()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Make sure models are loaded
    global tokenizer, aspect_encoders, aspect_models
    if tokenizer is None:
        initialize_models()
        
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    results = {}
    detected_keywords = {}
    
    for aspect, keywords in emotional_aspects.items():
        if aspect in aspect_models and aspect in aspect_encoders:
            # Preprocessing text to match keywords
            cleaned = clean_text(text.lower())
            processed = preprocess_text(cleaned)
            
            # Find detected keywords
            aspect_keywords = []
            for keyword in keywords:
                matches = re.findall(r'\b' + keyword + r'[a-z]*\b', processed)
                aspect_keywords.extend(matches)
            
            # Remove duplicates while preserving order
            aspect_keywords = list(dict.fromkeys(aspect_keywords))
            
            # Sentiment prediction
            sentiment = predict_aspect_sentiment(text, aspect)
            
            results[aspect] = sentiment
            detected_keywords[aspect] = aspect_keywords
    
    return jsonify({
        'sentiments': results,
        'keywords': detected_keywords
    })

# Initialize models when the app starts
with app.app_context():
    initialize_models()
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
