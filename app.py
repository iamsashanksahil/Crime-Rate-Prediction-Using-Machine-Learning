import os
import warnings
import pickle
import io
import base64

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

# Suppress sklearn warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

# CONFIG
CSV_FILE = 'NaviMumbai_Crime_Data_Updated.csv'
MODEL_FILE = 'crime_model.pkl'

CRIME_ENCODER = 'label_encoder_crime.pkl'
CITY_ENCODER = 'label_encoder_city.pkl'
GENDER_ENCODER = 'label_encoder_gender.pkl'
WEAPON_ENCODER = 'label_encoder_weapon.pkl'


# CREATE APP
app = Flask(__name__)
app.secret_key = os.environ.get(
    'SECRET_KEY',
    'navimumbai_crime_prediction_app'
)

# Demo credentials for academic project
users = {
    "admin": generate_password_hash("admin123")
}


# LOAD DATASET
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"{CSV_FILE} not found."
    )

df = pd.read_csv(CSV_FILE)


# DATA CLEANING
# Fill missing values
df['City'] = df['City'].fillna('Unknown')
df['Victim Gender'] = df['Victim Gender'].fillna('Unknown')
df['Weapon Used'] = df['Weapon Used'].fillna('Unknown')
df['Crime Description'] = df['Crime Description'].fillna('Other')

df['Victim Age'] = df['Victim Age'].fillna(0)
df['Longitude'] = df['Longitude'].fillna(0)
df['Latitude'] = df['Latitude'].fillna(0)

# Safe time conversion
df['Time of Occurrence'] = (
    df['Time of Occurrence']
    .astype(str)
    .fillna('00:00')
)

df['Time'] = pd.to_numeric(
    df['Time of Occurrence']
    .str.split(':')
    .str[0],
    errors='coerce'
)

df['Time'] = df['Time'].fillna(0)


# CREATE MODEL IF MISSING
if not os.path.exists(MODEL_FILE):
    print("Model files not found. Creating model...")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Create encoders
    le_crime = LabelEncoder()
    le_city = LabelEncoder()
    le_gender = LabelEncoder()
    le_weapon = LabelEncoder()

    # Encode columns
    df['Crime_Encoded'] = (
        le_crime.fit_transform(
            df['Crime Description']
        )
    )

    df['City_Encoded'] = (
        le_city.fit_transform(df['City'])
    )

    df['Gender_Encoded'] = (
        le_gender.fit_transform(
            df['Victim Gender']
        )
    )

    df['Weapon_Encoded'] = (
        le_weapon.fit_transform(
            df['Weapon Used']
        )
    )

    # Features
    X = df[
        [
            'Longitude',
            'Latitude',
            'Time',
            'City_Encoded',
            'Victim Age',
            'Gender_Encoded',
            'Weapon_Encoded'
        ]
    ].values

    y = df['Crime_Encoded'].values

    # Train model
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    with open(CRIME_ENCODER, 'wb') as f:
        pickle.dump(le_crime, f)

    with open(CITY_ENCODER, 'wb') as f:
        pickle.dump(le_city, f)

    with open(GENDER_ENCODER, 'wb') as f:
        pickle.dump(le_gender, f)

    with open(WEAPON_ENCODER, 'wb') as f:
        pickle.dump(le_weapon, f)

    print("Model created successfully.")


# LOAD MODEL + ENCODERS
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

with open(CRIME_ENCODER, 'rb') as f:
    le_crime = pickle.load(f)

with open(CITY_ENCODER, 'rb') as f:
    le_city = pickle.load(f)

with open(GENDER_ENCODER, 'rb') as f:
    le_gender = pickle.load(f)

with open(WEAPON_ENCODER, 'rb') as f:
    le_weapon = pickle.load(f)


# VISUALIZATION FUNCTIONS
def plot_to_base64():
    img = io.BytesIO()

    plt.savefig(
        img,
        format='png',
        bbox_inches='tight'
    )

    img.seek(0)

    plot_url = base64.b64encode(
        img.getvalue()
    ).decode('utf8')

    plt.close('all')

    return plot_url


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

    temp_df = df.copy()

    temp_df['Severity'] = (
        temp_df['Crime Description']
        .map(severity_scores)
    )

    crime_severity = (
        temp_df.groupby(
            'Crime Description'
        )['Severity']
        .mean()
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 8))

    if len(crime_severity) == 0:
        plt.text(
            0.5,
            0.5,
            'No data available',
            ha='center',
            va='center'
        )
        plt.axis('off')

    else:
        plt.pie(
            crime_severity,
            labels=crime_severity.index,
            autopct='%1.1f%%',
            startangle=90
        )

        plt.title(
            'Crime Severity Distribution'
        )

        plt.axis('equal')

    return plot_to_base64()


def create_gender_chart():

    gender_counts = (
        df['Victim Gender']
        .dropna()
        .value_counts()
    )

    plt.figure(figsize=(10, 6))

    if len(gender_counts) == 0:
        plt.text(
            0.5,
            0.5,
            'No data available',
            ha='center',
            va='center'
        )

        plt.axis('off')

    else:
        gender_counts.plot(
            kind='bar'
        )

        plt.title(
            'Number of Crimes by Victim Gender'
        )

        plt.xlabel('Gender')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=0)

    return plot_to_base64()


def create_crime_heatmap():

    crime_locations = (
        df.dropna(
            subset=[
                'Longitude',
                'Latitude'
            ]
        )
    )

    plt.figure(figsize=(12, 8))

    if len(crime_locations) == 0:
        plt.text(
            0.5,
            0.5,
            'No location data available',
            ha='center',
            va='center'
        )

        plt.axis('off')

    else:
        plt.hexbin(
            crime_locations['Longitude'],
            crime_locations['Latitude'],
            gridsize=30,
            cmap='YlOrRd'
        )

        plt.colorbar(
            label='Crime Density'
        )

        plt.title(
            'Crime Density Heatmap'
        )

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    return plot_to_base64()


# ROUTES
@app.route('/')
def index():

    if 'username' in session:
        return redirect(
            url_for('dashboard')
        )

    return redirect(
        url_for('login')
    )


@app.route(
    '/login',
    methods=['GET', 'POST']
)
def login():

    error = None

    if request.method == 'POST':

        username = (
            request.form['username']
            .strip()
        )

        password = (
            request.form['password']
        )

        if (
            username in users
            and
            check_password_hash(
                users[username],
                password
            )
        ):

            session['username'] = (
                username
            )

            return redirect(
                url_for(
                    'dashboard'
                )
            )

        error = (
            'Invalid credentials.'
        )

    return render_template(
        'login.html',
        error=error
    )


@app.route('/logout')
def logout():

    session.pop(
        'username',
        None
    )

    return redirect(
        url_for('login')
    )

@app.route('/dashboard')
def dashboard():

    if 'username' not in session:
        return redirect(
            url_for('login')
        )

    return render_template(
        'dashboard.html',

        crime_severity_chart=
        create_crime_severity_chart(),

        gender_chart=
        create_gender_chart(),

        crime_heatmap=
        create_crime_heatmap(),

        total_crimes=
        len(df),

        total_cities=
        df['City'].nunique(),

        total_crime_types=
        df['Crime Description'].nunique(),

        avg_age=
        round(
            df['Victim Age']
            .mean(),
            1
        )
    )

@app.route(
    '/predict',
    methods=['GET', 'POST']
)
def predict():

    if 'username' not in session:
        return redirect(
            url_for('login')
        )

    cities = sorted(
        df['City']
        .dropna()
        .unique()
    )

    genders = sorted(
        df['Victim Gender']
        .dropna()
        .unique()
    )

    weapons = sorted(
        df['Weapon Used']
        .dropna()
        .unique()
    )

    prediction_result = None
    probabilities = None

    if request.method == 'POST':

        try:

            longitude = float(
                request.form[
                    'longitude'
                ]
            )

            latitude = float(
                request.form[
                    'latitude'
                ]
            )

            city = request.form[
                'city'
            ]

            time = float(
                request.form[
                    'time'
                ]
            )

            age = int(
                request.form[
                    'age'
                ]
            )

            gender = request.form[
                'gender'
            ]

            weapon = request.form[
                'weapon'
            ]

            city_encoded = (
                le_city.transform(
                    [city]
                )[0]
            )

            gender_encoded = (
                le_gender.transform(
                    [gender]
                )[0]
            )

            weapon_encoded = (
                le_weapon.transform(
                    [weapon]
                )[0]
            )

            X = np.array([
                [
                    longitude,
                    latitude,
                    time,
                    city_encoded,
                    age,
                    gender_encoded,
                    weapon_encoded
                ]
            ])

            prediction = (
                model.predict(X)[0]
            )

            prediction_proba = (
                model.predict_proba(
                    X
                )[0]
            )

            prediction_result = (
                le_crime
                .inverse_transform(
                    [prediction]
                )[0]
            )

            crime_classes = (
                le_crime.classes_
            )

            probabilities = [
                (
                    crime,
                    round(
                        prob * 100,
                        2
                    )
                )
                for crime, prob
                in zip(
                    crime_classes,
                    prediction_proba
                )
            ]

            probabilities.sort(
                key=lambda x: x[1],
                reverse=True
            )

        except Exception as e:

            flash(
                f"Prediction Error: {e}"
            )

    return render_template(
        'predict.html',
        cities=cities,
        genders=genders,
        weapons=weapons,
        prediction=prediction_result,
        probabilities=probabilities
    )



if __name__ == '__main__':
    app.run(
        debug=True
    )
