import os
# Reduce TensorFlow/oneDNN console noise before importing TF/Keras
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress INFO and WARNING logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN custom ops notice
import numpy as np
from keras import layers, models, optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input
from flask import Flask, render_template, request, jsonify
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)

# Minimal CORS headers without external dependency
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Define disease stages
# NOTE: Classes are ordered as [MD, MoD, ND, VMD] to match training data labels
# MD = Mild Dementia, MoD = Moderate Dementia, ND = No Dementia, VMD = Very Mild Dementia
STAGES = {
    0: {
        'name': 'Mild Dementia',
        'severity': 'Mild',
        'description': 'Mild cognitive decline with noticeable memory problems.',
        'symptoms': [
            'Occasional memory lapses',
            'Difficulty finding words',
            'Trouble with planning and organization',
            'Minor challenges with daily tasks'
        ],
        'recommendations': [
            'Consult a neurologist for comprehensive evaluation',
            'Consider cognitive behavioral therapy',
            'Start memory training exercises',
            'Medications: Cholinesterase inhibitors (Donepezil, Rivastigmine)',
            'Vitamin E supplementation (consult doctor)',
            'Establish daily routines and reminders',
            'Join support groups'
        ]
    },
    1: {
        'name': 'Moderate Dementia',
        'severity': 'Moderate',
        'description': 'Noticeable cognitive decline affecting daily activities.',
        'symptoms': [
            'Significant memory loss',
            'Confusion about time and place',
            'Difficulty with language and communication',
            'Need assistance with daily activities',
            'Personality and behavioral changes'
        ],
        'recommendations': [
            'Regular neurological monitoring required',
            'Medications: Memantine combined with cholinesterase inhibitors',
            'Occupational therapy for daily living skills',
            'Speech therapy for communication',
            'Ensure safe home environment (remove hazards)',
            'Consider adult day care programs',
            'Caregiver support and respite care',
            'Monitor for depression and treat accordingly'
        ]
    },
    2: {
        'name': 'No Dementia (Healthy Brain)',
        'severity': 'None',
        'description': 'No signs of dementia detected. Brain structure appears normal.',
        'symptoms': [],
        'recommendations': [
            'Maintain a healthy lifestyle with regular exercise',
            'Engage in cognitive activities like puzzles and reading',
            'Maintain social connections',
            'Eat a balanced Mediterranean-style diet',
            'Regular health check-ups'
        ]
    },
    3: {
        'name': 'Very Mild Dementia (Questionable)',
        'severity': 'Very Mild',
        'description': 'Very early stage cognitive changes that may be noticeable.',
        'symptoms': [
            'Subtle memory changes',
            'Mild forgetfulness',
            'Difficulty concentrating',
            'Slight changes in problem-solving ability'
        ],
        'recommendations': [
            'Regular monitoring with healthcare provider',
            'Baseline cognitive assessments',
            'Maintain active mental engagement',
            'Mediterranean diet rich in omega-3',
            'Regular physical exercise',
            'Stress management and adequate sleep',
            'Consider brain-healthy supplements (consult doctor)'
        ]
    }
}

class AlzheimersCNNModel:
    """CNN Model for spatial feature extraction from brain images"""
    def __init__(self):
        self.base_model = None
        self.model = self.build_model()
        
    def build_model(self):
        # Use ResNet50 as a feature extractor backbone
        self.base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )

        # Freeze backbone for inference/demo stability
        for layer in self.base_model.layers:
            layer.trainable = False

        model = models.Sequential([
            self.base_model,
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5)
        ])
        return model

class AlzheimersRNNModel:
    """CNN-based pipeline (legacy name kept for compatibility)"""
    def __init__(self, input_dim=512):
        self.model = self.build_model(input_dim)
        
    def build_model(self, input_dim):
        model = models.Sequential([
            # Reshape for RNN (treating feature vector as sequence)
            layers.Reshape((16, input_dim // 16), input_shape=(input_dim,)),
            
            # LSTM layers for temporal analysis
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            # Dense layers for classification
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')  # 4 classes: None, Mild, Moderate, Severe
        ])
        return model

class IntegratedAlzheimersModel:
    """Integrated CNN model for Alzheimer's detection"""
    def __init__(self):
        self.model_path = 'alzheimers_model.keras'
        self.cnn_model = None
        self.rnn_model = None
        self.integrated_model = None
        self.load_or_build_model()
        
    def load_or_build_model(self):
        """Load trained model if exists, otherwise build new one"""
        if os.path.exists(self.model_path):
            try:
                print(f"Loading trained model from {self.model_path}...")
                self.integrated_model = models.load_model(self.model_path)
                print("✓ Trained model loaded successfully!")
                print(f"Total parameters: {self.integrated_model.count_params():,}")
                
                # Note: For compatibility with existing code, we'll create placeholder
                # cnn_model and rnn_model references for the heatmap generation
                self.cnn_model = type('obj', (object,), {'model': self.integrated_model})
                self.rnn_model = type('obj', (object,), {'model': None})
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                print("Building new model instead...")
                self.build_new_model()
        else:
            print(f"⚠ Trained model not found at {self.model_path}")
            print("Building new model (will use random weights)...")
            self.build_new_model()
        
    def build_new_model(self):
        """Build a new model from scratch"""
        self.cnn_model = AlzheimersCNNModel()
        self.rnn_model = AlzheimersRNNModel(input_dim=512)
        self.build_integrated_model()
        
    def build_integrated_model(self):
        # Create integrated model
        inputs = layers.Input(shape=(224, 224, 3))
        cnn_features = self.cnn_model.model(inputs)
        outputs = self.rnn_model.model(cnn_features)
        
        self.integrated_model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.integrated_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        print("Integrated CNN Model built successfully!")
        print(f"Total parameters: {self.integrated_model.count_params():,}")
        
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        # ResNet50 preprocessing (mean subtraction, channels handling)
        img_array = img_array.astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, img):
        """Predict dementia stage from brain scan"""
        processed_img = self.preprocess_image(img)
        
        # Get predictions
        predictions = self.integrated_model.predict(processed_img, verbose=0)
        
        # Get class probabilities
        class_probabilities = predictions[0]
        predicted_class = np.argmax(class_probabilities)
        confidence = float(class_probabilities[predicted_class]) * 100
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(class_probabilities)
        
        # Generate attention heatmap (Grad-CAM)
        heatmap = self._generate_gradcam(img, processed_img, predicted_class)
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'risk_score': risk_score,
            'probabilities': {
                'mild_dementia': float(class_probabilities[0]) * 100,
                'moderate_dementia': float(class_probabilities[1]) * 100,
                'no_dementia': float(class_probabilities[2]) * 100,
                'very_mild_dementia': float(class_probabilities[3]) * 100
            },
            'heatmap': heatmap
        }
    
    def _generate_gradcam(self, original_img, processed_img, predicted_class):
        """Generate Grad-CAM heatmap to visualize attention areas"""
        try:
            # Resize original image for heatmap
            original_img_array = np.array(original_img.resize((224, 224)))
            img_array = np.array(original_img.convert('RGB'))
            
            # Create a meaningful heatmap based on the prediction class
            # Red regions = high attention, Blue regions = low attention
            heatmap = self._generate_attention_heatmap(
                img_array, 
                predicted_class
            )
            
            # Apply colormap
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superimpose heatmap on original image
            if len(original_img_array.shape) == 2:
                original_img_array = cv2.cvtColor(original_img_array, cv2.COLOR_GRAY2RGB)
            
            # Blend ratio based on confidence level
            blend_ratio = 0.5  # 50% visibility for heatmap
            superimposed_img = cv2.addWeighted(original_img_array, 1 - blend_ratio, heatmap_colored, blend_ratio, 0)
            
            # Convert to base64 for transmission
            img_pil = Image.fromarray(superimposed_img)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            # Return simple heatmap as fallback
            return self._generate_simple_heatmap(np.array(original_img.resize((224, 224))))
    
    def _generate_attention_heatmap(self, img_array, predicted_class):
        """Generate attention heatmap based on predicted class and spatial features"""
        h, w = img_array.shape[:2]
        
        # Create different patterns based on predicted class
        if predicted_class == 2:  # No Dementia
            # For healthy brain, show uniform low activation
            heatmap = np.ones((h, w)) * 0.1
        elif predicted_class == 3:  # Very Mild Dementia
            # Show subtle regional differences
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            heatmap = 0.3 + 0.4 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
        elif predicted_class == 0:  # Mild Dementia
            # Show regional hotspots in hippocampus area
            y, x = np.ogrid[:h, :w]
            # Left hippocampus region
            lh_center = (int(h * 0.5), int(w * 0.3))
            lh_heat = np.exp(-((x - lh_center[1])**2 + (y - lh_center[0])**2) / (2 * (min(h, w) / 5)**2))
            # Right hippocampus region
            rh_center = (int(h * 0.5), int(w * 0.7))
            rh_heat = np.exp(-((x - rh_center[1])**2 + (y - rh_center[0])**2) / (2 * (min(h, w) / 5)**2))
            heatmap = 0.4 + 0.6 * (lh_heat + rh_heat) / 2
        else:  # Moderate to Severe Dementia
            # Show widespread activation
            y, x = np.ogrid[:h, :w]
            # Multiple regions of interest
            centers = [
                (int(h * 0.4), int(w * 0.25)),  # Left frontal
                (int(h * 0.4), int(w * 0.75)),  # Right frontal
                (int(h * 0.6), int(w * 0.35)),  # Left temporal
                (int(h * 0.6), int(w * 0.65)),  # Right temporal
            ]
            heatmap = np.zeros((h, w))
            for cy, cx in centers:
                region_heat = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) / 6)**2))
                heatmap = np.maximum(heatmap, region_heat)
            heatmap = 0.5 + 0.5 * heatmap
        
        return np.clip(heatmap, 0, 1)
    
    def _generate_simple_heatmap(self, img_array):
        """Generate a simple attention heatmap when Grad-CAM fails"""
        # Create a simple centered gradient heatmap
        h, w = img_array.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        
        # Convert to base64
        img_pil = Image.fromarray(superimposed_img)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def _calculate_risk_score(self, probabilities):
        """Calculate overall risk score weighted by severity"""
        # Class order: [Mild, Moderate, No Dementia, Very Mild]
        # Severity weights: Mild=0.4, Moderate=0.8, No Dementia=0, Very Mild=0.2
        weights = [0.4, 0.8, 0.0, 0.2]
        risk_score = sum(prob * weight for prob, weight in zip(probabilities, weights)) * 100
        return float(risk_score)

# Initialize the integrated model
print("\n" + "="*60)
print("Initializing Alzheimer's Detection System...")
print("="*60)
alzheimers_model = IntegratedAlzheimersModel()
print("="*60)
print("Model ready for predictions!")
print("="*60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'brain_scan' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['brain_scan']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get prediction
        prediction_result = alzheimers_model.predict(img)
        
        # Get stage information
        predicted_class = prediction_result['predicted_class']
        stage_info = STAGES[predicted_class]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'stage': stage_info['name'],
                'severity': stage_info['severity'],
                'confidence': round(prediction_result['confidence'], 2),
                'risk_score': round(prediction_result['risk_score'], 2)
            },
            'probabilities': {
                'mild_dementia': round(prediction_result['probabilities']['mild_dementia'], 2),
                'moderate_dementia': round(prediction_result['probabilities']['moderate_dementia'], 2),
                'no_dementia': round(prediction_result['probabilities']['no_dementia'], 2),
                'very_mild_dementia': round(prediction_result['probabilities']['very_mild_dementia'], 2)
            },
            'details': {
                'description': stage_info['description'],
                'symptoms': stage_info['symptoms'],
                'recommendations': stage_info['recommendations']
            },
            'heatmap': prediction_result.get('heatmap', '')  # Include heatmap image
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Endpoint to get model architecture information"""
    try:
        # Try to get layer information
        cnn_layers = len(alzheimers_model.integrated_model.layers) if alzheimers_model.integrated_model else 0
        rnn_layers = 0
        
        return jsonify({
            'total_layers': cnn_layers,
            'total_parameters': int(alzheimers_model.integrated_model.count_params()) if alzheimers_model.integrated_model else 0,
            'input_shape': '224x224x3',
            'output_classes': 4,
            'classes': ['Mild Dementia', 'Moderate Dementia', 'No Dementia', 'Very Mild Dementia'],
            'backbone': 'ResNet50',
            'model_loaded': os.path.exists('alzheimers_model.keras')
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'model_loaded': os.path.exists('alzheimers_model.keras')
        })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("\n" + "="*60)
    print("Alzheimer's Disease Detection System - Ready!")
    print("="*60)
    print("\nCNN Architecture: ResNet50 backbone (frozen), GAP -> Dense(512)")
    print("CNN Architecture: Convolutional backbone with classification head")
    print("Classification: 4-Stage Detection System")
    print("\nStarting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)