"""
Model Training Script for Facial Emotion Recognition

This script uses Transfer Learning with EfficientNet for high accuracy:
1. Loads the FER2013 dataset from the specified directory
2. Preprocesses images with data augmentation (resize to 224x224, normalize)
3. Builds a CNN using EfficientNetB0 pretrained on ImageNet (transfer learning)
4. Trains the model in two phases:
   - Phase 1: Frozen base model (15 epochs)
   - Phase 2: Fine-tuning with unfrozen base (15 epochs)
5. Evaluates the model on test data
6. Saves the trained model as 'face_emotionModel.h5'

The model will classify faces into 7 emotions:
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Expected Accuracy: 60-70% on FER2013 test set

Note: This script is designed to run on Google Colab with GPU for faster training.
      For local training, you may need to reduce epochs or use a smaller model.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback

# Path to the dataset (update this to your dataset location)
# For Google Colab: '/content/fer2013'
# For local: '/Users/mofiyinebo/Documents/Covenant University/Last Year/Alpha Semester/2025:2026 Notes/CSC415/Assignments/Models/archive'
DATASET_PATH = '/Users/mofiyinebo/Documents/Covenant University/Last Year/Alpha Semester/2025:2026 Notes/CSC415/Assignments/Models/archive'

# Training and test data paths
train_dir = os.path.join(DATASET_PATH, 'train')
test_dir = os.path.join(DATASET_PATH, 'test')

# Image parameters for EfficientNet
IMG_SIZE = 224  # EfficientNet uses 224x224 images
BATCH_SIZE = 32  # Reduced for memory efficiency
EPOCHS_PHASE1 = 15  # Phase 1: Frozen base
EPOCHS_PHASE2 = 15  # Phase 2: Fine-tuning

# Emotion labels (7 classes)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("=" * 60)
print("FACIAL EMOTION RECOGNITION - MODEL TRAINING")
print("=" * 60)
print(f"\nDataset Path: {DATASET_PATH}")
print(f"Training Data: {train_dir}")
print(f"Test Data: {test_dir}")
print(f"\nEmotion Classes: {emotion_labels}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)

# Data Augmentation for training data (more aggressive for better generalization)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,           # Normalize pixel values to [0, 1]
    rotation_range=15,           # Randomly rotate images ±15 degrees
    width_shift_range=0.15,      # Randomly shift images horizontally
    height_shift_range=0.15,     # Randomly shift images vertically
    shear_range=0.15,            # Shear transformation
    zoom_range=0.15,             # Randomly zoom images
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest',         # Fill missing pixels after transformations
    validation_split=0.2         # Use 20% of training data for validation
)

# Only rescaling for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training data (with validation split)
print("\n[1/6] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',  # EfficientNet requires RGB images
    class_mode='categorical',
    subset='training',  # Use training subset
    shuffle=True
)

# Load validation data
print("[2/6] Loading validation data...")
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',  # Use validation subset
    shuffle=False
)

# Load test data
print("[3/6] Loading test data...")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")

# Build the model using Transfer Learning with EfficientNet
print("\n[4/6] Building the model with EfficientNet (Transfer Learning)...")

# Load EfficientNetB0 pretrained on ImageNet
base_model = EfficientNetB0(
    include_top=False,  # Don't include classification head
    weights='imagenet',  # Use ImageNet pretrained weights
    input_shape=(IMG_SIZE, IMG_SIZE, 3)  # RGB images
)

# Freeze the base model initially (for Phase 1 training)
base_model.trainable = False

# Build the complete model
model = Sequential([
    base_model,  # EfficientNet base
    GlobalAveragePooling2D(),  # Global pooling instead of flatten
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Callbacks for training
print("\n[5/6] Setting up training callbacks...")

# Save the best model during training
checkpoint = ModelCheckpoint(
    'face_emotionModel.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,  # Only save the best model
    verbose=1
)

# Stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,  # Stop after 8 epochs without improvement
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce LR by half
    patience=4,  # Wait 4 epochs before reducing
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# PHASE 1: Train with frozen base model
print("\n[6/6] Training the model (Phase 1: Frozen Base)...")
print("=" * 60)
print("Training with frozen EfficientNet base for feature extraction")
print(f"Epochs: {EPOCHS_PHASE1}")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# PHASE 2: Fine-tune with unfrozen base model
print("\n" + "=" * 60)
print("Training Phase 2: Fine-tuning with unfrozen base")
print("=" * 60)

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Epochs: {EPOCHS_PHASE2}")
print("=" * 60)

# Continue training with unfrozen base
history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model on test data
print("\n[7/7] Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\nModel saved as: face_emotionModel.h5")
print("=" * 60)

# Combine training history from both phases
all_accuracy = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
all_val_accuracy = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
all_loss = history_phase1.history['loss'] + history_phase2.history['loss']
all_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

# Display training history summary
print("\nTraining History Summary:")
print(f"Best Training Accuracy: {max(all_accuracy):.4f}")
print(f"Best Validation Accuracy: {max(all_val_accuracy):.4f}")
print(f"Final Training Loss: {all_loss[-1]:.4f}")
print(f"Final Validation Loss: {all_val_loss[-1]:.4f}")
print("\n" + "=" * 60)
print("Model Architecture: EfficientNetB0 + Custom Head")
print("Training Strategy: Transfer Learning (2-phase)")
print("  - Phase 1: Frozen base (15 epochs)")
print("  - Phase 2: Fine-tuning (15 epochs)")
print("Dataset: FER2013 (35,887 images)")
print("Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")
print("=" * 60)
print("\n✅ You can now use this model in your Flask application!")
print("💡 To run the Flask app: python3 app.py")
print("🌐 Then open: http://localhost:5000")
