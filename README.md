# Alzheimer's Disease Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**An AI-powered web application for early detection and staging of Alzheimer's disease using deep learning and brain MRI analysis.**

[Features](#-features) • [Technologies](#-technologies) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

This project implements a comprehensive deep learning system for detecting and classifying Alzheimer's disease stages from brain MRI scans. The system uses a state-of-the-art ResNet50-based architecture to analyze brain images and classify them into four categories:

- **No Dementia** - Healthy brain with no signs of cognitive decline
- **Very Mild Dementia** - Early stage with subtle cognitive changes
- **Mild Dementia** - Noticeable cognitive decline with memory problems
- **Moderate Dementia** - Significant cognitive decline affecting daily activities

## ✨ Features

### 🧠 Deep Learning Pipeline
- **Transfer Learning**: ResNet50 backbone pre-trained on ImageNet
- **Custom Architecture**: Dense layers with batch normalization and dropout
- **Attention Visualization**: Grad-CAM heatmaps showing brain regions of interest
- **Risk Scoring**: Intelligent risk assessment algorithm

### 🎯 Classification Capabilities
- 4-class dementia staging system
- Confidence scores for each prediction
- Detailed symptom analysis
- Personalized medical recommendations

### 🌐 Web Interface
- Modern, responsive UI design
- Real-time image upload and analysis
- Interactive heatmap visualization
- Detailed results dashboard
- Mobile-friendly interface

## 🛠️ Technologies

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | TensorFlow 2.16, Keras 3.4 |
| **Backend** | Flask 3.0.0 |
| **Computer Vision** | OpenCV 4.8, PIL |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Data Processing** | NumPy, Pandas |

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/alzheimers-detection.git
cd alzheimers-detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Files

Make sure you have the trained model files:
- `alzheimers_model.keras` (required)
- `alzheimers_weights.keras` (optional)

If these files are not present, the application will build a new model with random weights.

## 🚀 Usage

### Running the Application

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Or manually:**
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Training the Model

To train the model with your own data:

```bash
python train_model.py
```

**Note**: Ensure you have properly formatted data in the `images/` directory:
```
images/
├── train/       # Training images
├── valid/       # Validation images
├── test/        # Test images
└── CSV_datafiles/
    ├── _train_classes.csv
    ├── _valid_classes.csv
    └── _test_classes.csv
```

### Data Format

CSV files should contain:
- `filename`: Image filename
- `MD`: Mild Dementia (0 or 1)
- `MoD`: Moderate Dementia (0 or 1)
- `ND`: No Dementia (0 or 1)
- `VMD`: Very Mild Dementia (0 or 1)

## 🏗️ Architecture

### Model Architecture

```
Input: 224×224×3 RGB Image
    ↓
ResNet50 Backbone (frozen)
    ↓
Global Average Pooling → 2048 features
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.2)
    ↓
Dense(4) + Softmax
    ↓
Output: [Mild, Moderate, No Dementia, Very Mild]
```

### Key Components

1. **Feature Extraction**: ResNet50 extracts spatial features from brain images
2. **Classification Head**: Custom dense layers for disease staging
3. **Attention Mechanism**: Grad-CAM for visualization
4. **Risk Assessment**: Weighted probability scoring

### Training Configuration

- **Image Size**: 224×224×3
- **Batch Size**: 32
- **Epochs**: 20
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, AUC

## 📊 Results

The model provides:
- **Stage Classification**: One of 4 dementia stages
- **Confidence Score**: Prediction confidence percentage
- **Risk Score**: Weighted risk assessment (0-100)
- **Class Probabilities**: Individual class likelihoods
- **Visual Heatmap**: Brain region attention visualization

## 🧪 API Endpoints

### POST `/analyze`
Analyze a brain MRI scan.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `brain_scan` (image file)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "stage": "Mild Dementia",
    "severity": "Mild",
    "confidence": 87.5,
    "risk_score": 35.2
  },
  "probabilities": {
    "mild_dementia": 87.5,
    "moderate_dementia": 5.2,
    "no_dementia": 2.1,
    "very_mild_dementia": 5.2
  },
  "details": {
    "description": "...",
    "symptoms": [...],
    "recommendations": [...]
  },
  "heatmap": "base64_encoded_image..."
}
```

### GET `/model-info`
Get model architecture information.

**Response:**
```json
{
  "total_layers": 175,
  "total_parameters": 25000000,
  "input_shape": "224x224x3",
  "output_classes": 4,
  "backbone": "ResNet50",
  "model_loaded": true
}
```

## 📁 Project Structure

```
alzheimers-detection/
├── app.py                  # Flask application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Frontend interface
├── static/
│   └── style.css          # Stylesheet
└── images/
    ├── train/             # Training data
    ├── valid/             # Validation data
    ├── test/              # Test data
    └── CSV_datafiles/     # Class labels
```

## ⚠️ Important Disclaimers

1. **Medical Disclaimer**: This tool is for research and educational purposes only. It is NOT a substitute for professional medical diagnosis, advice, or treatment.

2. **Clinical Use**: DO NOT use this system for making actual medical decisions. Always consult qualified healthcare professionals.

3. **Data Privacy**: Ensure compliance with HIPAA and data protection regulations when handling medical images.

4. **Model Limitations**: The model's accuracy depends on training data quality and may not generalize to all populations.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent framework
- Flask team for the web framework
- Medical imaging research community
- Open-source machine learning community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ for medical research and education**

⭐ Star this repo if you find it helpful!

</div>
