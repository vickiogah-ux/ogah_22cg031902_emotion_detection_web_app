"""
Flask Web Application for Facial Emotion Detection

This application:
1. Serves an HTML form where users can input their information and upload a photo
2. Processes the uploaded image using the trained emotion detection model
3. Detects the person's emotion from the photo (using 48x48 grayscale CNN model)
4. Saves user information and image to an SQLite database
5. Returns a personalized message based on the detected emotion

Model: Basic CNN trained on 48x48 grayscale images
"""

import os
import sqlite3
import base64
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'face_emotionModel.h5'
model = None

try:
    print(f"Attempting to load model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("⚠️ Model not loaded - predictions will return 'unknown'")

# Emotion labels matching the training data
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Emotion responses - personalized messages for each emotion
EMOTION_RESPONSES = {
    'angry': "You look angry. Take a deep breath - everything will be okay! 😤",
    'disgust': "You seem disgusted. What's bothering you? 🤢",
    'fear': "You appear fearful. Don't worry, you're safe here! 😨",
    'happy': "You're smiling! Keep spreading that positive energy! 😊",
    'sad': "You are frowning. Why are you sad? Don't worry, things will get better! 😢",
    'surprise': "You look surprised! What happened? 😮",
    'neutral': "You have a neutral expression. Having a calm day? 😐"
}

# Database initialization
def init_db():
    """
    Initialize the SQLite database and create the users table if it doesn't exist.
    
    Table schema:
    - id: Auto-incrementing primary key
    - name: User's name
    - email: User's email
    - age: User's age
    - image_data: Binary data of the uploaded image
    - detected_emotion: The emotion detected by the model
    - timestamp: When the record was created
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            image_data BLOB NOT NULL,
            detected_emotion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Initialize database when app starts
init_db()

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction.
    
    Steps:
    1. Load the image using OpenCV
    2. Convert to grayscale (model expects grayscale)
    3. Detect face using Haar Cascade (optional but improves accuracy)
    4. Resize to 48x48 pixels (model input size)
    5. Normalize pixel values to [0, 1]
    6. Reshape to match model input shape (1, 48, 48, 1)
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not load image: {image_path}")
            # Return dummy array if image fails to load
            return np.zeros((1, 48, 48, 1), dtype=np.float32)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to detect face using Haar Cascade (with faster settings)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,  # Faster detection
            minNeighbors=3,   # Less strict
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If face is detected, crop to face region
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use the first detected face
            gray = gray[y:y+h, x:x+w]
        
        # Resize to 48x48 (model input size)
        img_resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Reshape to (1, 48, 48, 1) for model input
        img_reshaped = img_normalized.reshape(1, 48, 48, 1)
        
        return img_reshaped
        
    except Exception as e:
        print(f"❌ Error in preprocess_image: {e}")
        return np.zeros((1, 48, 48, 1), dtype=np.float32)

def predict_emotion(image_path):
    """
    Predict the emotion from an image using the trained model.
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    if model is None:
        print("❌ Model is None - cannot predict")
        return "unknown", 0.0
    
    try:
        print(f"Preprocessing image: {image_path}")
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        print(f"Processed image shape: {processed_image.shape}")
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        print(f"Predictions: {predictions[0]}")
        
        # Get the emotion with highest probability
        emotion_index = np.argmax(predictions[0])
        confidence = predictions[0][emotion_index]
        emotion_label = EMOTION_LABELS[emotion_index]
        
        print(f"✅ Predicted: {emotion_label} (confidence: {confidence:.2%})")
        return emotion_label, float(confidence)
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return "unknown", 0.0

def save_to_database(name, email, age, image_path, emotion):
    """
    Save user information and image to the database.
    
    Args:
        name: User's name
        email: User's email
        age: User's age
        image_path: Path to the uploaded image
        emotion: Detected emotion
    """
    # Read image as binary data
    with open(image_path, 'rb') as file:
        image_data = file.read()
    
    # Insert into database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO users (name, email, age, image_data, detected_emotion)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, email, age, image_data, emotion))
    
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route that handles both GET and POST requests.
    
    GET: Display the form
    POST: Process form submission, detect emotion, save to database
    """
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        image = request.files.get('image')
        
        # Validate inputs
        if not name or not email or not age or not image:
            return render_template('index.html', 
                                 error="Please fill in all fields and upload an image.",
                                 result=None)
        
        # Validate image file
        if image.filename == '':
            return render_template('index.html', 
                                 error="Please select an image file.",
                                 result=None)
        
        # Save uploaded image temporarily
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        
        # Predict emotion
        emotion, confidence = predict_emotion(image_path)
        
        # Get personalized response
        response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        
        # Save to database
        try:
            save_to_database(name, email, int(age), image_path, emotion)
        except Exception as e:
            print(f"Error saving to database: {e}")
            return render_template('index.html', 
                                 error="Error saving data to database.",
                                 result=None)
        
        # Prepare result
        result = {
            'name': name,
            'emotion': emotion.capitalize(),
            'confidence': f"{confidence * 100:.2f}%",
            'message': response_message
        }
        
        return render_template('index.html', error=None, result=result)
    
    # GET request - just show the form
    return render_template('index.html', error=None, result=None)

@app.route('/health')
def health():
    """Health check endpoint for deployment services like Render."""
    return {'status': 'healthy', 'model_loaded': model is not None}

if __name__ == '__main__':
    # Run the Flask app
    # For local testing: debug=True
    # For production: debug=False
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
