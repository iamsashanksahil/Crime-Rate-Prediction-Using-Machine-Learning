# Navi Mumbai Crime Prediction System

## Overview

The **Navi Mumbai Crime Prediction System** is a machine learning based web application developed to analyze crime trends and predict probable crime categories in Navi Mumbai. The system uses historical crime data and multiple predictive factors such as geographical location, victim demographics, time of occurrence, and suspected weapon information to estimate the most likely crime type.

The application also includes an interactive dashboard for crime analytics, helping users visualize crime severity, victim demographics, and geographical crime density.

**Note:**  
The trained machine learning model (`crime_model.pkl`) and label encoder files are intentionally excluded from this repository because of GitHub file size limitations. They are automatically generated the first time the application is executed.

---


# Project Evolution & Enhancements

## Initial Project Version

The initial version of the **Navi Mumbai Crime Prediction System** was developed as an academic machine learning project with a functional Flask backend and crime prediction logic.

The system initially supported:

* Login authentication
* Machine learning-based crime prediction
* Dashboard visualizations
* Basic HTML templates
* Crime severity charts
* Victim gender analytics
* Crime density heatmap

Although functional, the initial implementation had several technical and usability limitations.

---

## Issues Identified in Initial Version

### 1. Machine Learning Model Compatibility Issues

#### Problem

The project generated warnings during model loading due to differences in Scikit-learn versions used while training and running the model.

Issues observed:

* `InconsistentVersionWarning`
* Model loading instability
* Version compatibility warnings

#### Fix Implemented

Compatibility handling was added in `app.py`:

```python
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)
```

This ensured smoother execution without unnecessary compatibility warnings.

---

### 2. Prediction Form Reset Problem

#### Problem

After predicting a crime category, all prediction input fields automatically reset to default values.

This caused:

* Repeated manual input
* Poor user experience
* Difficulty comparing multiple predictions

Affected fields:

* Longitude
* Latitude
* City
* Time
* Victim age
* Victim gender
* Weapon selection

#### Fix Implemented

The prediction form was improved using:

```python
request.form.get()
```

This preserved user-entered values even after prediction.

Example:

```html
value="{{ request.form.get('longitude', '73.0295') }}"
```

Result:

* No repeated input
* Better usability
* Faster experimentation with predictions

---

### 3. Prediction Page UI Issues (`predict.html`)

#### Problems in Earlier UI

The original prediction page had:

* Basic visual styling
* Limited responsiveness
* Poor spacing
* Weak prediction result presentation
* Less attractive interface

#### Enhancements Made

The page was redesigned with:

* Better layout structure
* Improved spacing and responsiveness
* Cleaner visual hierarchy
* Styled probability progress bars
* Better prediction result presentation
* Enhanced card-based design

Additional improvements:

* Prediction result section redesigned
* Probability distribution made easier to understand
* Better Bootstrap integration

---

### 4. Dashboard Improvements (`dashboard.html`)

#### Earlier Problems

Dashboard UI appeared visually basic and lacked better data presentation.

#### Improvements Made

Enhanced:

* Layout structure
* Dashboard responsiveness
* Graph presentation
* Visual consistency

Dashboard analytics improved with:

* Crime Severity Distribution
* Gender-wise Crime Analysis
* Crime Density Heatmap
* Better statistical presentation

Result:

* More professional dashboard appearance
* Better readability
* Improved analytics experience

---

### 5. Base Layout Enhancements (`base.html`)

#### Previous Limitations

The earlier layout had:

* Basic navigation styling
* Limited visual consistency
* Simple page structure

#### Improvements Made

Enhanced:

* Navbar styling
* Theme consistency
* Layout spacing
* Better user interface flow

Result:

* Cleaner application structure
* Improved visual consistency

---

### 6. Login Page Improvements (`login.html`)

#### Earlier Issues

* Basic UI
* Limited styling
* Less professional appearance

#### Improvements Made

Enhanced:

* Better alignment
* Improved form styling
* Cleaner authentication interface
* More responsive structure

Result:

* Better first impression
* More polished interface

---

### 7. Machine Learning Prediction Behavior

#### Observation

Since the model is trained on historical crime data, predictions depend entirely on statistical patterns learned from the dataset.

In some cases, predictions may appear unexpected due to:

* Dataset imbalance
* Rare feature combinations
* Historical pattern dependencies
* Feature weighting during model training

Example:

```text
Weapon Used: Explosive
Predicted Crime: Fraud
```

#### Explanation

The machine learning model does not apply human reasoning. It predicts outcomes based on patterns present in historical data used during training.

Therefore, predictions depend on learned relationships between features such as:

* Location
* Time
* Victim demographics
* Weapon used
* Historical crime distribution

#### Future Improvements Considered

Potential enhancements for better prediction consistency:

* Dataset cleaning
* Feature refinement
* Removal of unrealistic combinations
* Better data preprocessing
* Improved training dataset quality


---


## Final Enhanced Version

The final version of the project now includes:

### Backend Improvements

* Better model compatibility handling
* Cleaner Flask routing
* Improved prediction logic
* Better preprocessing support

### Frontend Improvements

* Enhanced dashboard UI
* Improved prediction page
* Responsive layouts
* Better visual hierarchy
* Cleaner Bootstrap integration

### User Experience Improvements

* Form persistence after prediction
* Better prediction visualization
* Improved usability
* Cleaner navigation flow

### Machine Learning Features

* Random Forest Classifier
* Probability-based predictions
* Encoded categorical variables
* Crime category prediction

---

## Before vs After Improvements

| Feature                  | Initial Version        | Final Version              |
| ------------------------ | ---------------------- | -------------------------- |
| Prediction Form          | Reset after prediction | Retains values             |
| UI Design                | Basic                  | Enhanced                   |
| Dashboard                | Simple                 | Improved                   |
| Prediction Visualization | Basic                  | Better probability display |
| Login Page               | Minimal                | Cleaner UI                 |
| HTML Templates           | Basic structure        | Enhanced responsiveness    |
| Model Handling           | Warning issues         | Better compatibility       |

---

## Project Outcome

The project evolved from a basic academic Flask machine learning system into a more polished, visually enhanced, and user-friendly crime prediction application with improved usability, prediction handling, and dashboard presentation.


---


## Features

### 1. Secure Login System

* Session-based authentication
* Password protected access
* Admin login support

**Default Credentials**

```text
Username: admin
Password: admin123
```

---

### 2. Interactive Analytics Dashboard

The dashboard provides crime-related visualizations and statistical insights.

#### Dashboard Features

* **Total Crimes Analysis**
  Displays total recorded crime entries

* **Cities Covered**
  Shows number of city areas included in the dataset

* **Crime Categories Overview**
  Displays available crime classifications

* **Average Victim Age Analysis**

* **Crime Severity Distribution**
  Pie chart showing severity level of different crimes

* **Victim Gender Analysis**
  Visual breakdown of crimes by victim gender

* **Crime Density Heatmap**
  Displays crime concentration using geographic coordinates

---

### 3. Crime Prediction System

The prediction system estimates the **most probable crime type** based on:

* Longitude
* Latitude
* City
* Time of occurrence
* Victim age
* Victim gender
* Weapon used

After prediction, the system displays:

* Predicted crime category
* Probability distribution for all crime types
* Visual confidence bars for easier interpretation

---

## Technology Stack

### Backend

* Python
* Flask

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* Bootstrap Icons
* Jinja2 Templates

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

---

## Machine Learning Model

### Algorithm Used

**Random Forest Classifier**

The model was trained using historical Navi Mumbai crime data.

### Features Used for Prediction

1. Longitude
2. Latitude
3. Time of occurrence
4. City
5. Victim age
6. Victim gender
7. Weapon used

### Data Preprocessing

The system performs:

* Missing value handling
* Label encoding of categorical variables
* Time conversion to numeric format
* Feature extraction and preprocessing

---

## Installation Guide

### Prerequisites

Install:

* **Python 3.12.x Recommended**
* pip package manager

**Avoid Python 3.14+** because some ML packages may not yet be fully compatible.

---

### Step 1: Clone or Download Project

```bash
git clone https://github.com/iamsashanksahil/navimumbai-crime-prediction.git
cd navimumbai-crime-prediction
```

---

### Step 2: Create Virtual Environment (Recommended)

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Mac/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

If pip gives issues:

```bash
python -m pip install -r requirements.txt
```

---

### Step 4: Ensure Dataset Exists

Place this dataset in project root:

```text
NaviMumbai_Crime_Data_Updated.csv
```

---

### Step 5: Run Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## Automatic Model Generation

This project does not include the trained model (`crime_model.pkl`) or label encoder files in the repository because they are generated automatically.

When the application is run for the first time:

- The dataset is loaded.
- Data preprocessing is performed.
- Label encoders are created.
- A Random Forest model is trained.
- The trained model and encoder files are saved locally.

This process only happens once. Future runs will load the generated files automatically.

---

## Project Structure

```text
navimumbai-crime-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── NaviMumbai_Crime_Data_Updated.csv
│
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── login.html
│   └── predict.html
│
├── static/
│
└── (Model and encoder files are generated automatically on first run.)
```

---

## Dataset Description

The dataset contains historical crime records with:

* Report Number
* Date Reported
* Date of Occurrence
* Time of Occurrence
* City
* Crime Code
* Crime Description
* Longitude
* Latitude
* Victim Age
* Victim Gender
* Weapon Used

---

## Prediction Workflow

1. User logs in
2. User enters prediction inputs
3. Data gets encoded using label encoders
4. Random Forest model processes input
5. Predicted crime type is generated
6. Probability distribution is displayed

---

## Troubleshooting

### 1. `pip not recognized`

Run:

```powershell
python -m pip install -r requirements.txt
```

---

### 2. Model Version Errors

Delete the generated model files:

```powershell
del crime_model.pkl
del label_encoder_*.pkl
```

Then rerun the application:

```powershell
python app.py
```

The application will automatically recreate the Random Forest model and label encoders from the dataset.

---

### 3. Python Compatibility Issues

Recommended version:

```text
Python 3.12.x
```

If using Python 3.14 and getting package issues:

Install Python 3.12 and recreate virtual environment.

---

## Future Enhancements

* Real-time crime monitoring
* GIS-based map integration
* Advanced predictive analytics
* Multi-user authentication
* Admin analytics panel
* Crime trend forecasting

---

## Disclaimer

This project is developed for **educational and academic purposes only**. Predictions generated by the system should not be treated as real-world law enforcement decisions.

---

## Project Information

**Project Title:** Crime Rate Prediction Using Machine Learning

**Developed By:** Sashank Sahil

**Course:** B.Tech Computer Science Engineering

**Academic Project Year:** Final Year Project
