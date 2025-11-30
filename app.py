import os
import warnings
import numpy as np
import pandas as pd
import pickle
import io
import base64
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
#for suppressing if any sklearn warnings recieved
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Check if model files exist, if not create them
if not os.path.exists('crime_model.pkl'):
    print("Model files not found. Creating them now...")
    # Load data
    df = pd.read_csv('NaviMumbai_Crime_Data_Updated.csv')

    # Data preprocessing
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Convert time to numeric format for model
    df['Time'] = pd.to_numeric(df['Time of Occurrence'].str.split(':').str[0], errors='coerce')

    # Create encoders
    le_crime = LabelEncoder()
    le_city = LabelEncoder()
    le_gender = LabelEncoder()
    le_weapon = LabelEncoder()

    # Fit encoders
    df['Crime_Encoded'] = le_crime.fit_transform(df['Crime Description'])
    df['City_Encoded'] = le_city.fit_transform(df['City'])
    df['Gender_Encoded'] = le_gender.fit_transform(df['Victim Gender'])
    df['Weapon_Encoded'] = le_weapon.fit_transform(df['Weapon Used'])

    # Prepare features for model
    X = df[['Longitude', 'Latitude', 'Time', 'City_Encoded', 'Victim Age', 'Gender_Encoded', 'Weapon_Encoded']].values
    y = df['Crime_Encoded'].values

    # Train a Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    with open('crime_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoder_crime.pkl', 'wb') as f:
        pickle.dump(le_crime, f)

    with open('label_encoder_city.pkl', 'wb') as f:
        pickle.dump(le_city, f)

    with open('label_encoder_gender.pkl', 'wb') as f:
        pickle.dump(le_gender, f)

    with open('label_encoder_weapon.pkl', 'wb') as f:
        pickle.dump(le_weapon, f)

    print("Model files created successfully.")

# Create Flask app
app = Flask(__name__)
app.secret_key = 'navimumbai_crime_prediction_app'

# Load the model and encoders
with open('crime_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder_crime.pkl', 'rb') as f:
    le_crime = pickle.load(f)

with open('label_encoder_city.pkl', 'rb') as f:
    le_city = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

with open('label_encoder_weapon.pkl', 'rb') as f:
    le_weapon = pickle.load(f)


df = pd.read_csv('NaviMumbai_Crime_Data_Updated.csv')

# Mock user database 
users = {
    "admin": generate_password_hash("admin123")
}

# Define helper functions for visualizations
def create_crime_severity_chart():

    severity_scores = {
        "Murder": 10,
        "Robbery": 8,
        "Rape": 9,
        "Assault": 7,
        "Theft": 5,
        "Fraud": 6,
        "Drug-related crime": 6,
        "Domestic violence": 7,
        "Cybercrime": 6,
        "Kidnapping": 8,
        "Other": 4
    }


    df['Severity'] = df['Crime Description'].map(severity_scores)

    crime_severity = df.groupby('Crime Description')['Severity'].mean().dropna().sort_values(ascending=False)

    # Create pie chart
    plt.figure(figsize=(10, 8))
    if len(crime_severity) == 0:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.axis('off')
    else:
        plt.pie(crime_severity, labels=crime_severity.index, autopct='%1.1f%%', startangle=90)
        plt.title('Crime Severity Distribution')
        plt.axis('equal')

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url

def create_gender_chart():
    gender_counts = df['Victim Gender'].dropna().value_counts()

    # Create bar chart
    plt.figure(figsize=(10, 6))

    if len(gender_counts) == 0:
        plt.text(0.5, 0.5, 'No gender data available', ha='center', va='center')
        plt.axis('off')
    else:
        gender_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Crimes by Victim Gender')
        plt.xlabel('Gender')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=0)

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url

def create_crime_heatmap():

    crime_locations = df.dropna(subset=['Longitude', 'Latitude'])


    plt.figure(figsize=(12, 8))

    if len(crime_locations) == 0:
        plt.text(0.5, 0.5, 'No location data available', ha='center', va='center')
        plt.axis('off')
    else:
        # Create heatmap using hexbin
        plt.hexbin(crime_locations['Longitude'], crime_locations['Latitude'],
                  gridsize=30, cmap='YlOrRd')
        plt.colorbar(label='Crime Density')

        plt.title('Crime Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url

# Define Flask routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Generate plots for dashboard
    crime_severity_chart = create_crime_severity_chart()
    gender_chart = create_gender_chart()
    crime_heatmap = create_crime_heatmap()

    return render_template('dashboard.html',
                          crime_severity_chart=crime_severity_chart,
                          gender_chart=gender_chart,
                          crime_heatmap=crime_heatmap)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    cities = sorted([str(city) for city in df['City'].dropna().unique()])
    genders = sorted([str(gender) for gender in df['Victim Gender'].dropna().unique()])

    weapons = sorted([str(weapon) for weapon in df['Weapon Used'].dropna().unique()])

    prediction_result = None
    probabilities = None

    if request.method == 'POST':
        # Get form data
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        city = request.form['city']
        time = float(request.form['time'])
        age = int(request.form['age'])
        gender = request.form['gender']
        weapon = request.form['weapon']

        # Encode categorical features
        city_encoded = le_city.transform([city])[0]
        gender_encoded = le_gender.transform([gender])[0]
        weapon_encoded = le_weapon.transform([weapon])[0]

        # Create input array for prediction
        X = np.array([[longitude, latitude, time, city_encoded, age, gender_encoded, weapon_encoded]])

        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]

        # Decode prediction
        prediction_result = le_crime.inverse_transform([prediction])[0]

        # Get probabilities for all classes
        crime_classes = le_crime.classes_
        probabilities = [(crime, round(prob * 100, 2)) for crime, prob in zip(crime_classes, prediction_proba)]
        probabilities.sort(key=lambda x: x[1], reverse=True)

    return render_template('predict.html',
                          cities=cities,
                          genders=genders,
                          weapons=weapons,
                          prediction=prediction_result,
                          probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)