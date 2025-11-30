# Navi Mumbai Crime Prediction System

## Overview

A comprehensive web application for analyzing and predicting crime patterns in Navi Mumbai using machine learning. This system helps law enforcement agencies and urban planners identify potential crime hotspots and understand crime distribution based on various factors.

## Features

### 1. Secure Authentication
- Login system with password protection  
- Session management for secure access  
- Default credentials: **admin/admin123**  

### 2. Interactive Dashboard
- **Crime Severity Distribution:** Pie chart visualizing the severity levels of different crime types  
- **Gender Analysis:** Bar graph showing crime distribution by victim gender  
- **Crime Density Map:** Heat map displaying concentration of crimes across Navi Mumbai  

### 3. Predictive Analytics
- ML-powered crime type prediction based on:
  - Geographic coordinates (longitude/latitude)
  - Location (city area)
  - Time of day
  - Victim demographics (age, gender)
  - Weapon information
- Probability distribution for different crime types

## Technical Implementation

### Technology Stack
- **Backend:** Flask (Python web framework)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, Bootstrap 5

### Machine Learning Model
- **Algorithm:** Random Forest Classifier
- **Features:** Location, time, victim characteristics, and weapon data
- **Preprocessing:** Label encoding for categorical variables
- **Data Handling:** Robust handling of missing values and outliers

## Setup Instructions

### Prerequisites
- Python 3.8+ (**3.12.7 recommended**)
- pip package manager

### Installation

Clone the repository or download the project files:
```bash
git clone https://github.com/yourusername/navimumbai-crime-prediction.git
cd navimumbai-crime-prediction
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure the crime dataset is in the project directory:
```
NaviMumbai_Crime_Data_Updated.csv
```

Run the application:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

Login with default credentials:
```
Username: admin
Password: admin123
```

## Troubleshooting Common Setup Issues

### Package Installation Problems
If you encounter errors during package installation, especially with Python 3.12, try:
```bash
pip install --upgrade pip setuptools wheel
pip install Flask numpy pandas matplotlib seaborn scikit-learn Werkzeug
```

### Model File Errors
If you see scikit-learn version warnings, regenerate the model files:
```bash
rm crime_model.pkl label_encoder_crime.pkl label_encoder_city.pkl label_encoder_gender.pkl label_encoder_weapon.pkl
```
Then restart the application to rebuild them with your current scikit-learn version.

### Matplotlib Errors
If you encounter Matplotlib errors, ensure you're using the Agg backend by adding this at the top of `app.py`:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## Project Structure
```
navimumbai-crime-prediction/
│
├── app.py                     # Main Flask application
├── NaviMumbai_Crime_Data_Updated.csv   # Crime dataset
│
├── templates/                 # HTML templates
│   ├── base.html              # Base template
│   ├── login.html             # Login page
│   ├── dashboard.html         # Dashboard with visualizations
│   └── predict.html           # Crime prediction interface
│
├── crime_model.pkl            # Trained model
├── label_encoder_crime.pkl    # Label encoder for crime types
├── label_encoder_city.pkl     # Label encoder for cities
├── label_encoder_gender.pkl   # Label encoder for genders
├── label_encoder_weapon.pkl   # Label encoder for weapons
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Data Description

The system uses a comprehensive dataset of Navi Mumbai crime records containing:
- **Report Number:** Unique identifier for each crime report
- **Date Reported:** When the crime was reported to authorities
- **Date of Occurrence:** When the crime actually occurred
- **Time of Occurrence:** Time of day when the crime occurred
- **City:** Area within Navi Mumbai where the crime happened
- **Crime Code:** Numerical code representing the crime type
- **Crime Description:** Type of crime (Assault, Theft, Fraud, etc.)
- **Longitude/Latitude:** Geographic coordinates of the crime location
- **Victim Age:** Age of the victim
- **Victim Gender:** Gender of the victim
- **Weapon Used:** Type of weapon used in the crime

## Code Implementation Details

### Data Preprocessing
The application includes automatic preprocessing of the crime data:
- Converting time to numeric format
- Encoding categorical variables (crime type, city, gender, weapon)
- Handling missing values in visualizations and model training
- Feature selection for prediction

### Visualization Generation
The dashboard creates visualizations on-demand using:
- Crime severity scores mapped to different crime types
- Gender-based crime distribution analysis
- Geographic density mapping using Matplotlib's hexbin

### Prediction Pipeline
The prediction system follows these steps:
1. User inputs location and other parameters
2. Data is encoded using saved label encoders
3. Random Forest model predicts the crime type
4. Probability distribution for all crime types is calculated
5. Results are displayed in an easy-to-understand format

### Model Performance
The Random Forest model analyzes multiple features to predict crime types. The most important features in prediction order:
1. Geographic location (longitude/latitude)
2. Time of day
3. Victim age
4. City area

## Future Improvements
- User management system with different access levels
- Real-time crime data integration
- Advanced time-series forecasting
- Mobile application interface
- Enhanced reporting and analytics features
- Integration with GIS mapping services
- Improved model performance with ensemble methods

## Submission Information
- **Project Title:** Navi Mumbai Crime Prediction System
- **Author:** [Your Name]
- **Date:** March 27, 2025
- **Course/Module:** [Course Name]
- **Institution:** [Your Institution]

## Credits and License
This project was developed as a demonstration of predictive analytics in law enforcement. The data used is for educational purposes only.

This project was developed for educational purposes. The predictions made by this system should not be used as the sole basis for law enforcement decisions without proper validation and domain expertise.
