# Navi Mumbai Crime Prediction System

## Overview

The **Navi Mumbai Crime Prediction System** is a machine learning based web application developed to analyze crime trends and predict probable crime categories in Navi Mumbai. The system uses historical crime data and multiple predictive factors such as geographical location, victim demographics, time of occurrence, and suspected weapon information to estimate the most likely crime type.

The application also includes an interactive dashboard for crime analytics, helping users visualize crime severity, victim demographics, and geographical crime density.

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

## Project Structure

```text
navimumbai-crime-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── NaviMumbai_Crime_Data_Updated.csv
│
├── crime_model.pkl
├── label_encoder_crime.pkl
├── label_encoder_city.pkl
├── label_encoder_gender.pkl
├── label_encoder_weapon.pkl
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   └── predict.html
│
└── static/
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

Delete model files:

```powershell
del crime_model.pkl
del label_encoder_*.pkl
```

Then rerun:

```powershell
python app.py
```

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
**Academic Project Year:** Final Year Project (Updation)
