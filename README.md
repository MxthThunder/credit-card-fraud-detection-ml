# 💳 Credit Card Fraud Detection using Machine Learning

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

A complete **machine learning project** for detecting fraudulent credit card transactions using **Logistic Regression** and **Random Forest** classifiers. This project follows the methodology from the [Intellipaat ML tutorial](https://www.youtube.com/watch?v=jCoF1rMs_0s) with a production-ready Flask web application and interactive dashboard.

## 🎯 Project Overview

Credit card fraud is a critical problem in the financial sector. This system uses **PCA-transformed transaction features** to classify transactions as legitimate or fraudulent with high accuracy while handling severe class imbalance.

### Key Features
- ✅ **Two ML Models**: Logistic Regression + Random Forest (Claude-1x class models)
- ✅ **SMOTE**: Handles class imbalance (~0.17% fraud rate)
- ✅ **Flask REST API**: Production-ready backend with `/predict` and `/batch_predict`
- ✅ **Interactive Dashboard**: Real-time fraud detection with probability scores
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Sample Data Generator**: Synthetic dataset for testing without Kaggle download
- ✅ **Complete Documentation**: Setup, API reference, and usage examples

## 📂 Project Structure

```
credit-card-fraud-detection-ml/
├── app.py                     # Flask web application
├── train.py                   # ML training pipeline (5 steps)
├── generate_sample_data.py    # Synthetic dataset generator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Python gitignore
├── templates/
│   └── index.html            # Interactive web dashboard
├── models/                    # Saved models (generated after training)
│   ├── fraud_model.pkl
│   └── scaler.pkl
└── data/                      # Dataset directory
    └── creditcard.csv        # Kaggle dataset or synthetic data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/MxthThunder/credit-card-fraud-detection-ml.git
cd credit-card-fraud-detection-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get the Dataset

**Option 1**: Download from Kaggle (recommended for production)
1. Go to [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` folder

**Option 2**: Generate synthetic data (for testing)
```bash
python generate_sample_data.py
```

### Train the Model

```bash
python train.py
```

**Output**:
- Loads and explores the dataset
- Applies SMOTE to balance classes
- Trains Logistic Regression and Random Forest
- Evaluates both models
- Saves the best model to `models/fraud_model.pkl`

**Expected Results** (on real Kaggle dataset):
- **Precision**: ~90%+
- **Recall**: ~80%+
- **F1-Score**: ~85%+
- **ROC-AUC**: ~95%+

### Run the Web Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

## 🌐 API Endpoints

### 1. **Health Check**
```bash
GET /health
```
Response:
```json
{"status": "running", "model_loaded": true}
```

### 2. **Single Prediction**
```bash
POST /predict
Content-Type: application/json

{
  "features": [406.0, -3.04, 2.54, -2.54, 1.37, -0.96, 0.46, -0.82, 0.35, -0.58, 0.09, -2.26, 2.32, -0.49, 1.55, 0.43, -0.55, -0.14, 0.24, -0.28, 0.22, 0.13, -0.11, 0.09, -0.18, 0.36, 0.01, -0.02, 0.05, 529.0]
}
```
Response:
```json
{
  "prediction": 1,
  "label": "FRAUD",
  "fraud_probability": 98.45,
  "legitimate_probability": 1.55,
  "risk_level": "HIGH"
}
```

### 3. **Batch Prediction**
```bash
POST /batch_predict
Content-Type: application/json

{
  "records": [
    {"features": [...]},
    {"features": [...]}
  ]
}
```

### 4. **Model Information**
```bash
GET /model_info
```

## 📊 Training Pipeline (5 Steps)

The `train.py` follows a comprehensive 5-step process:

1. **Load & Explore**: Dataset statistics, class distribution, feature analysis
2. **Preprocess**: StandardScaler for Time and Amount (V1-V28 already PCA-scaled)
3. **Handle Imbalance**: SMOTE oversampling to balance classes
4. **Train Models**: Logistic Regression + Random Forest with hyperparameters
5. **Evaluate**: Metrics, confusion matrix, classification report → Save best model

## 🧪 Testing

### Using the Web Dashboard
1. Start the Flask app: `python app.py`
2. Open `http://localhost:5000`
3. Click "Load Fraud Sample" or "Load Legit Sample"
4. Click "Analyze Transaction"
5. View real-time prediction with probability scores

### Using curl
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [406, -3.04, 2.54, -2.54, 1.37, -0.96, 0.46, -0.82, 0.35, -0.58, 0.09, -2.26, 2.32, -0.49, 1.55, 0.43, -0.55, -0.14, 0.24, -0.28, 0.22, 0.13, -0.11, 0.09, -0.18, 0.36, 0.01, -0.02, 0.05, 529]}'
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core language
- **scikit-learn**: ML models (Logistic Regression, Random Forest)
- **imbalanced-learn**: SMOTE for class imbalance
- **Flask**: Web framework & REST API
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Visualization (in notebooks)

## 📈 Model Performance

| Metric | Logistic Regression | Random Forest | 
|--------|--------------------|--------------|
| **Accuracy** | ~99.9% | ~99.9% |
| **Precision** | ~88% | ~92% |
| **Recall** | ~62% | ~84% |
| **F1-Score** | ~73% | ~88% |
| **ROC-AUC** | ~92% | ~97% |

*Results based on Kaggle dataset with SMOTE*

**Best Model**: Random Forest (selected by F1-Score)

## 🎓 Learning Resources

- **Tutorial Reference**: [Intellipaat - Credit Card Fraud Detection](https://www.youtube.com/watch?v=jCoF1rMs_0s)
- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **SMOTE**: [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- **Flask**: [Flask Official Docs](https://flask.palletsprojects.com/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**MxthThunder**
- GitHub: [@MxthThunder](https://github.com/MxthThunder)
- Project: [credit-card-fraud-detection-ml](https://github.com/MxthThunder/credit-card-fraud-detection-ml)

## ⭐ Acknowledgments

- Intellipaat for the comprehensive ML tutorial
- Kaggle for providing the credit card fraud dataset
- ULB (Université Libre de Bruxelles) for the original dataset research

---

**Built with ❤️ for learning Machine Learning and production deployment**
