import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from keras import layers, models, optimizers, callbacks
from keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
BASE_DIR = 'images'
TRAIN_CSV = os.path.join(BASE_DIR, 'CSV_datafiles', '_train_classes.csv')
VALID_CSV = os.path.join(BASE_DIR, 'CSV_datafiles', '_valid_classes.csv')
TEST_CSV = os.path.join(BASE_DIR, 'CSV_datafiles', '_test_classes.csv')

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Model save path
MODEL_SAVE_PATH = 'alzheimers_model.keras'
WEIGHTS_SAVE_PATH = 'alzheimers_weights.keras'

def load_data_from_csv(csv_path, image_dir):
    """Load data from CSV file and return images and labels"""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        filename = row['filename']
        filepath = os.path.join(image_dir, filename)
        
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                img = img.convert('RGB')
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img)
                
                # Create label from one-hot encoding [MD, MoD, ND, VMD]
                label = [row['MD'], row['MoD'], row['ND'], row['VMD']]
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        else:
            print(f"File not found: {filepath}")
    
    return np.array(images), np.array(labels)

def preprocess_images(images):
    """Preprocess images for ResNet50"""
    images = images.astype('float32')
    images = preprocess_input(images)
    return images

def create_data_generator(images, labels, batch_size=32, shuffle_data=True):
    """Create a data generator for training"""
    num_samples = len(images)
    
    while True:
        if shuffle_data:
            images, labels = shuffle(images, labels)
        
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Preprocess images
            batch_images = preprocess_images(batch_images)
            
            yield batch_images, batch_labels

def build_model():
    """Build the ResNet model for Alzheimer's classification"""
    # Load pre-trained ResNet50 (without top)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Freeze the first layers to keep pretrained knowledge
    # We'll unfreeze some layers later for fine-tuning
    for layer in base_model.layers[:len(base_model.layers)-10]:
        layer.trainable = False
    
    # Allow the last 10 layers to be trainable
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    # Build the model
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    
    # Add custom classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer for 4 classes
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model():
    """Main training function"""
    print("Loading training data...")
    train_images, train_labels = load_data_from_csv(TRAIN_CSV, TRAIN_DIR)
    print(f"Loaded {len(train_images)} training samples")
    
    print("Loading validation data...")
    valid_images, valid_labels = load_data_from_csv(VALID_CSV, VALID_DIR)
    print(f"Loaded {len(valid_images)} validation samples")
    
    print("Loading test data...")
    test_images, test_labels = load_data_from_csv(TEST_CSV, TEST_DIR)
    print(f"Loaded {len(test_images)} test samples")
    
    # Preprocess images
    print("Preprocessing images...")
    train_images = preprocess_images(train_images)
    valid_images = preprocess_images(valid_images)
    test_images = preprocess_images(test_images)
    
    # Build model
    print("Building model...")
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Create data generators
    train_gen = create_data_generator(train_images, train_labels, BATCH_SIZE, shuffle_data=True)
    valid_gen = create_data_generator(valid_images, valid_labels, BATCH_SIZE, shuffle_data=False)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_images) // BATCH_SIZE
    validation_steps = len(valid_images) // BATCH_SIZE
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            WEIGHTS_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the complete model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_gen = create_data_generator(test_images, test_labels, BATCH_SIZE, shuffle_data=False)
    test_steps = len(test_images) // BATCH_SIZE
    
    test_results = model.evaluate(test_gen, steps=test_steps, verbose=1)
    print(f"Test Loss: {test_results[0]}")
    print(f"Test Accuracy: {test_results[1]}")
    print(f"Test AUC: {test_results[2]}")
    
    # Save training history
    np.save('training_history.npy', history.history)
    print("Training history saved to training_history.npy")
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Best weights saved to: {WEIGHTS_SAVE_PATH}")

if __name__ == '__main__':
    train_model()

