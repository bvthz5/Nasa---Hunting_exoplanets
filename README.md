# NASA Exoplanet Classification System

A sophisticated web application for exoplanet classification developed for the NASA Space Apps Challenge 2025. This system enables both researchers and novices to explore exoplanet data using advanced machine learning models.

## 🚀 Features

### Core Functionality
- **Secure Authentication**: Researcher login system with session management
- **Multi-Dataset Support**: K2, TESS, and Kepler Objects of Interest (KOI) datasets
- **Real-time Classification**: Manual input for individual exoplanet analysis
- **Batch Processing**: CSV file upload for bulk classification
- **Model Retraining**: Hyperparameter tuning and model optimization
- **Professional UI**: NASA-themed responsive design with accessibility features

### Technical Features
- **Secure Authentication**: Password hashing, session management, and access control
- **Modular Architecture**: Clean separation of concerns with dedicated preprocessing utilities
- **Production-Ready**: Comprehensive logging, error handling, and monitoring
- **Accessibility**: WCAG 2.1 compliant with screen reader support and keyboard navigation
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Real-time Analytics**: Chart.js integration for data visualization

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** with Flask 3.0.3
- **XGBoost** for machine learning models
- **scikit-learn** for preprocessing and validation
- **pandas & numpy** for data manipulation
- **Flask-CORS** for cross-origin support

### Frontend
- **HTML5** with semantic markup
- **Tailwind CSS** for styling
- **Vanilla JavaScript** for interactivity
- **Chart.js** for data visualization
- **Font Awesome** for icons

### Development Tools
- **Type hints** throughout Python code
- **Comprehensive logging** with rotation
- **Error handling** with detailed JSON responses
- **Model versioning** with timestamps

## 📁 Project Structure

```
nasa-exoplanet-classifier/
├── app.py                          # Main Flask application
├── src/
│   └── utils/
│       └── preprocessing.py        # Modular preprocessing utilities
├── models/                         # Pre-trained model files
│   ├── k2/                        # K2 mission models
│   ├── tess/                      # TESS mission models
│   └── koi/                       # KOI models
├── templates/
│   └── index.html                 # Main application template
├── static/
│   ├── css/
│   │   └── styles.css             # Custom NASA-themed styles
│   └── js/
│       └── script.js              # Frontend JavaScript
├── logs/                          # Application logs
├── uploads/                       # Temporary file uploads
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nasa-exoplanet-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`
   
   **Login Credentials:**
   - Username: `user`
   - Password: `123`

## 📊 API Endpoints

### Authentication Endpoints

#### `GET /login`
Displays the login page for researchers.

#### `POST /login`
Authenticates users with username and password.

**Form Data:**
- `username`: Researcher username
- `password`: User password

#### `GET /logout`
Logs out the current user and redirects to login page.

### Core Endpoints

#### `GET /` (Protected)
Serves the main application interface (requires authentication).

#### `GET /stats?dataset={k2|tess|koi}` (Protected)
Returns model statistics for a specific dataset.

**Response:**
```json
{
  "dataset": "K2",
  "accuracy": 0.87,
  "class_distribution": {
    "CONFIRMED": 800,
    "CANDIDATE": 1200,
    "FALSE POSITIVE": 400
  },
  "features": ["pl_orbper", "pl_trandep", "st_teff"],
  "model_type": "XGBoost",
  "total_samples": 2400
}
```

#### `POST /predict` (Protected)
Predicts exoplanet class from JSON input.

**Request:**
```json
{
  "dataset": "k2",
  "pl_orbper": 1.7575,
  "pl_trandep": 0.0744,
  "st_teff": 4759
}
```

**Response:**
```json
{
  "prediction": "CANDIDATE",
  "confidence": 0.85,
  "probabilities": [0.1, 0.8, 0.1],
  "dataset": "K2",
  "features": {
    "pl_orbper": 1.7575,
    "pl_trandep": 0.0744,
    "st_teff": 4759
  }
}
```

#### `POST /upload` (Protected)
Handles CSV file upload for bulk classification.

**Form Data:**
- `file`: CSV file with exoplanet data
- `dataset`: Dataset name (k2, tess, koi)
- `retrain`: Boolean flag for model retraining

#### `POST /retrain` (Protected)
Retrains model with new hyperparameters.

**Request:**
```json
{
  "dataset": "k2",
  "n_estimators": 300,
  "max_depth": 10,
  "learning_rate": 0.2,
  "data": "base64_encoded_csv"  // optional
}
```

### Monitoring Endpoints

#### `GET /health`
Health check endpoint for monitoring.

#### `GET /system-status`
Detailed system status information.

## 🎯 Usage Guide

### Manual Classification
1. Select a dataset from the dropdown (K2, TESS, or KOI)
2. Enter exoplanet parameters:
   - **Orbital Period**: Time for one complete orbit (days)
   - **Transit Depth**: Fractional decrease in stellar brightness (ppm)
   - **Stellar Temperature**: Effective temperature of host star (Kelvin)
3. Click "Launch Classification" to get predictions

### Batch Processing
1. Prepare a CSV file with columns: `pl_orbper`, `pl_trandep`, `st_teff`
2. Select the appropriate dataset
3. Optionally check "Retrain model" for model improvement
4. Upload the file and view results

### Model Optimization
1. Adjust hyperparameters using the sliders:
   - **N Estimators**: 200-400 (number of boosting rounds)
   - **Max Depth**: 7-12 (maximum tree depth)
   - **Learning Rate**: 0.1-0.3 (step size shrinkage)
2. Click "Retrain Model" to optimize performance

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Model Paths
Models are expected in the following structure:
```
models/
├── k2/
│   ├── k2_xgboost_model.pkl
│   ├── k2_scaler.npy
│   ├── k2_imputer.pkl
│   └── k2_label_encoder.npy
├── tess/
│   └── [similar structure]
└── koi/
    └── [similar structure]
```

## 🎨 Customization

### NASA Theme Colors
The application uses a custom NASA color palette:
- **NASA Blue**: `#0B3D91`
- **NASA Red**: `#FC3D21`
- **NASA Dark**: `#1a1a2e`
- **NASA Light**: `#16213e`

### Accessibility Features
- High contrast mode toggle
- Screen reader announcements
- Keyboard navigation support
- Reduced motion preferences
- ARIA labels and descriptions

## 🧪 Testing

### Manual Testing
1. **Authentication Testing**:
   - Test login with correct credentials (`user`/`123`)
   - Test login with incorrect credentials
   - Verify logout functionality
   - Check session timeout behavior

2. **Application Testing**:
   - Test all three datasets (K2, TESS, KOI)
   - Verify form validation and error handling
   - Test file upload with various CSV formats
   - Check accessibility features with screen readers

### API Testing
Use tools like Postman or curl to test API endpoints:

```bash
# Test authentication
curl -X POST http://localhost:5000/login \
  -d "username=user&password=123" \
  -c cookies.txt

# Test prediction endpoint (with authentication)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"dataset": "k2", "pl_orbper": 1.7575, "pl_trandep": 0.0744, "st_teff": 4759}'

# Test stats endpoint (with authentication)
curl http://localhost:5000/stats?dataset=k2 \
  -b cookies.txt

# Test logout
curl http://localhost:5000/logout \
  -b cookies.txt
```

### Automated Testing
Run the comprehensive test suite:

```bash
# Test authentication system
python test_auth.py

# Test complete system
python test_system.py
```

## 🚀 Deployment

### Production Deployment
1. Set `FLASK_ENV=production`
2. Use a production WSGI server (e.g., Gunicorn)
3. Configure reverse proxy (e.g., Nginx)
4. Set up SSL certificates
5. Configure logging and monitoring

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📈 Performance Considerations

- **Model Caching**: Models are loaded once and cached in memory
- **File Size Limits**: 16MB maximum upload size
- **Log Rotation**: Automatic log file rotation to prevent disk space issues
- **Error Handling**: Graceful degradation when models are not available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed for the NASA Space Apps Challenge 2025. Please refer to the challenge guidelines for usage terms.

## 🏆 NASA Space Apps Challenge

This project was developed for the NASA Space Apps Challenge 2025, focusing on:
- **Data Exploration**: Enabling researchers to explore exoplanet datasets
- **Model Retraining**: Allowing continuous improvement of classification models
- **Accessibility**: Making space science accessible to everyone
- **Education**: Providing an intuitive interface for learning about exoplanets

## 📞 Support

For questions or issues:
1. Check the logs in the `logs/` directory
2. Review the API documentation above
3. Test with the health check endpoint: `GET /health`
4. Check system status: `GET /system-status`

---

**Team**: syntax_in_orbit  
**NASA Space Apps Challenge 2025**  
*Exploring the cosmos through technology*