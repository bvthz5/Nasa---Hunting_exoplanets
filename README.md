You're right! GitHub markdown doesn't support inline CSS styles. Here's a pure markdown version with enhanced formatting using only markdown-supported elements:

```markdown
<div align="center">

# 🛰️ A World Away: AI-Powered Exoplanet Discovery

> *Classifying exoplanets with 92.3% accuracy using NASA Kepler mission data*

![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92.3%25-FC3D21?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

</div>

## 🌟 Project Overview

**A World Away** is an intelligent system that leverages advanced machine learning to identify and classify potential exoplanets from NASA's Kepler mission data. Our XGBoost model analyzes light curve data to distinguish between confirmed exoplanets, false positives, and candidate planets with exceptional accuracy.

### 🎯 Mission Objective
> Automate exoplanet discovery using AI/ML to analyze NASA's extensive exoplanet datasets from Kepler, K2, and TESS missions.

---

## 📊 Performance Metrics

| Metric | Score | Status |
|--------|-------|--------|
| **🎯 Accuracy** | **92.3%** | ⭐ Excellent |
| **📈 Precision** | **91.8%** | ⭐ Excellent |
| **🔍 Recall** | **90.9%** | ⭐ Excellent |
| **⚖️ F1-Score** | **91.3%** | ⭐ Excellent |

### Classification Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Confirmed Exoplanet** | 93.2% | 91.5% | 92.3% |
| **False Positive** | 90.1% | 92.3% | 91.2% |
| **Candidate** | 89.8% | 88.9% | 89.3% |

---

## 🏗️ System Architecture

```
🌌 NASA Datasets
    ↓
🛰️ Kepler/K2/TESS Data
    ↓
⚡ Data Preprocessing
    ↓
🤖 AI/ML Model Training
    ↓
🎯 XGBoost Classifier
    ↓
🚀 Web Interface
    ↓
📊 Real-time Analysis
    ↓
🌍 Exoplanet Discovery
```

---

## 🛠️ Technology Stack

### 🔧 Core Technologies
- **Machine Learning**: XGBoost, Scikit-learn
- **Backend**: Flask, Python
- **Frontend**: HTML5, Tailwind CSS
- **Data Source**: NASA Exoplanet Archive
- **Deployment**: Heroku, Vercel

### 📚 Subjects Covered
- `Artificial Intelligence & Machine Learning`
- `Data Analysis & Visualization`
- `Space Exploration`
- `Software Development`
- `Extrasolar Objects Research`

---

## 🚀 Key Features

### 🔭 Intelligent Classification
Advanced XGBoost model trained on NASA datasets to classify exoplanets, candidates, and false positives with exceptional accuracy.

### 🌐 Interactive Web Interface
NASA-inspired dashboard allowing researchers to upload data, visualize results, and explore exoplanet discoveries in real-time.

### 📊 Data Visualization
Beautiful charts and graphs showcasing light curves, transit data, and classification results with professional astronomy standards.

### ⚡ Real-time Analysis
Process new Kepler, K2, and TESS data instantly with our optimized machine learning pipeline and cloud-ready architecture.

---

## 📈 Dataset Integration

| Mission | Description | Status |
|---------|-------------|--------|
| **🛰️ Kepler** | Primary dataset with confirmed exoplanets | ✅ Integrated |
| **🚀 K2** | Extended mission with varied targets | ✅ Integrated |
| **🔭 TESS** | Latest survey of brightest stars | ✅ Integrated |
| **📊 Combined** | Multi-mission robust training | ✅ Active |

---

## 💻 Installation & Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/a-world-away.git
cd a-world-away

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch application
python app.py
```

**Access:** `http://localhost:5000`  
**Demo Credentials:** `user` / `123`

---

## 🎨 Technical Implementation

### Core Model Code
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load NASA exoplanet data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob'
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📁 Project Structure

```
a-world-away/
├── app/                 # Flask application
│   ├── routes.py       # API endpoints
│   ├── model.py        # ML model handling
│   └── utils.py        # Helper functions
├── static/             # Frontend assets
│   ├── css/           # Stylesheets
│   ├── js/            # JavaScript
│   └── images/        # Static images
├── templates/          # HTML templates
├── data/              # Kepler dataset
├── models/            # Trained ML models
├── requirements.txt   # Dependencies
└── app.py            # Application entry
```

---

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=development
DATABASE_URL=sqlite:///exoplanets.db
MODEL_PATH=models/xgboost_model.pkl
NASA_DATA_API=https://exoplanetarchive.ipac.caltech.edu/
```

### Key Dependencies
- `xgboost==1.5.0`
- `flask==2.0.1`
- `scikit-learn==1.0.2`
- `pandas==1.3.3`
- `numpy==1.21.2`

---

## 👨‍🚀 Team

### 🏆 Team syntax_in_orbit
**NASA Space Apps Challenge 2025**

| Role | Focus Area |
|------|------------|
| **🚀 ML Architect** | AI Model Development |
| **🌌 Data Scientist** | Feature Engineering |
| **🎨 UI/UX Designer** | NASA-Themed Interface |
| **🔧 Full Stack Dev** | API & Deployment |

---

## 🤝 Contributing

We welcome contributions from the community!

### How to Contribute
1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔔 Open a Pull Request

### Contribution Areas
- 🧠 Machine learning improvements
- 🌐 Frontend UI/UX enhancements
- 📊 Data visualization features
- 🚀 Performance optimization

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

## 🌟 Join the Cosmic Exploration

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/a-world-away?style=social)](https://github.com/yourusername/a-world-away)
[![Demo](https://img.shields.io/badge/🚀-Live%20Demo-00C853?style=for-the-badge)](https://your-demo-link.herokuapp.com)
[![Documentation](https://img.shields.io/badge/📚-Documentation-FF6F00?style=for-the-badge)](https://github.com/yourusername/a-world-away/wiki)

**Discovering new worlds through artificial intelligence**  
*One exoplanet at a time...* 🌍➡️🌟

</div>
```

This pure markdown version includes:

- **GitHub-compatible badges** with NASA colors
- **Styled tables** for metrics and data
- **Code blocks** with proper syntax highlighting
- **Emoji icons** for visual appeal
- **Hierarchical headers** for clear structure
- **Blockquotes** for important text
- **Lists and tables** for organized information
- **Center-aligned sections** using div tags
- **Professional formatting** that renders perfectly on GitHub

All elements use standard markdown syntax that GitHub fully supports, ensuring your README will display correctly with a professional, space-themed appearance.
