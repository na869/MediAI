import os
import pickle
import io
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash, session, request, send_from_directory, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField, FloatField, SelectMultipleField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from fpdf import FPDF 

app = Flask(__name__)
app.secret_key = os.environ.get('MED_AI_SESSION_SECRET', 'MED-AI-INSTITUTIONAL-KEY-2026')

if os.environ.get('VERCEL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/users.db'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
db = SQLAlchemy(app)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'Capture.JPG', mimetype='image/vnd.microsoft.icon')

# ----------------------------
# Database Models
# ----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    predictions = db.relationship('Prediction', backref='patient_ref', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    patient_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    bp = db.Column(db.String(10))
    cholesterol = db.Column(db.String(10))
    na_to_k = db.Column(db.Float)
    condition = db.Column(db.String(100))
    predicted_drug = db.Column(db.String(100))
    confidence = db.Column(db.Float)

try:
    with app.app_context():
        db.create_all()
except Exception:
    pass

# ----------------------------
# ML Components
# ----------------------------
base_dir = os.path.abspath(os.path.dirname(__file__))
df_orig = pd.read_csv(os.path.join(base_dir, 'drug200.csv'))
df_orig.columns = df_orig.columns.str.lower()
df_orig.dropna(inplace=True)
df_orig.drop_duplicates(inplace=True)

sex_map = {'F': 0, 'M': 1}
bp_map = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}
chol_map = {'NORMAL': 0, 'HIGH': 1}

le_reason = LabelEncoder()
le_reason.fit(df_orig['reason'])
le_drug = LabelEncoder()
le_drug.fit(df_orig['drug'])
scaler = StandardScaler()
scaler.fit(df_orig[['age', 'na_to_k']])

X = df_orig.drop('drug', axis=1)
y = le_drug.transform(df_orig['drug'])
X['sex'] = X['sex'].map(sex_map)
X['bp'] = X['bp'].map(bp_map)
X['cholesterol'] = X['cholesterol'].map(chol_map)
X['reason'] = le_reason.transform(X['reason'])
X[['age', 'na_to_k']] = scaler.transform(X[['age', 'na_to_k']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42).fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# ----------------------------
# Logic & Utility
# ----------------------------
def get_aligned_probs(model, features, master_classes):
    if model is None:
        return np.zeros(len(master_classes))
    probs = model.predict_proba(features)[0]
    model_classes = model.classes_
    full_probs = np.zeros(len(master_classes))
    for i, class_idx in enumerate(model_classes):
        if class_idx < len(master_classes):
            full_probs[class_idx] = probs[i]
    return full_probs

MODEL_PROFILES = {
    'Random Forest': {'desc': 'Ensemble method using 100+ decision trees.', 'rationale': 'Probabilistic consensus weighting.', 'strength': 'High Stability.'},
    'Decision Tree': {'desc': 'Rule-based hierarchical classifier.', 'rationale': 'Entropy reduction logic.', 'strength': 'Interpretability.'},
    'KNN': {'desc': 'Spatial distance-based classifier.', 'rationale': 'Weighted Euclidean neighbors.', 'strength': 'Pattern Matching.'}
}

REGIMEN_DATA = {
    'Depression': {'dosage': '20mg / Day', 'frequency': 'Once daily', 'duration': '6 months+', 'precautions': 'Avoid alcohol.', 'remedies': 'CBT, Sunlight.', 'notes': 'Serotonin modulation.'},
    'Acne': {'dosage': 'Topical application', 'frequency': 'Nightly', 'duration': '12 weeks', 'precautions': 'Use SPF.', 'remedies': 'Low-glycemic diet.', 'notes': 'Sebum control.'},
    'Urinary Tract Infection': {'dosage': '100mg', 'frequency': 'Twice daily', 'duration': '5 days', 'precautions': 'Complete course.', 'remedies': 'Hydration.', 'notes': 'Antibacterial.'},
    'Asthma': {'dosage': '2 Puffs', 'frequency': 'PRN (As needed)', 'duration': 'Maintenance', 'precautions': 'Avoid triggers.', 'remedies': 'Air purification.', 'notes': 'Bronchodilation.'},
    'Anxiety': {'dosage': '0.5mg', 'frequency': 'Before bed', 'duration': '2-4 weeks', 'precautions': 'Dependency risk.', 'remedies': 'Therapy.', 'notes': 'GABA support.'},
    'Viral Infection': {'dosage': '500mg', 'frequency': 'Every 6 hours', 'duration': '3-5 days', 'precautions': 'Stay hydrated.', 'remedies': 'Rest, fluids.', 'notes': 'Symptomatic relief.'},
    'General Condition': {'dosage': 'Standard', 'frequency': 'Daily', 'duration': '7 days', 'precautions': 'Monitor vitals.', 'remedies': 'Rest.', 'notes': 'Non-specific stress.'}
}

SYMPTOM_MAP = {
    'low_mood': 'Depression', 'sadness': 'Depression', 'insomnia': 'Depression',
    'acne': 'Acne', 'rash': 'Acne',
    'chronic_pain': 'Pain', 'back_pain': 'Pain',
    'frequent_urination': 'Urinary Tract Infection', 'abdominal_pain': 'Urinary Tract Infection',
    'cough': 'Cough', 'sneeze': 'Cough', 'sore_throat': 'Cough',
    'short_breath': 'Asthma', 'chest_pain': 'Asthma', 'palpitations': 'Asthma',
    'anxiety': 'Anxiety', 'tremor': 'Anxiety', 'dizziness': 'Anxiety',
    'fever': 'Viral Infection', 'nausea': 'Viral Infection', 'headache': 'Viral Infection', 
    'fatigue': 'Viral Infection', 'vomiting': 'Viral Infection', 'diarrhea': 'Viral Infection',
    'muscle_weakness': 'General Condition', 'blurred_vision': 'General Condition', 
    'weight_loss': 'General Condition', 'night_sweats': 'General Condition', 'numbness': 'General Condition'
}

def infer_condition(symptoms):
    matched = [SYMPTOM_MAP.get(s) for r in symptoms if (s := r.lower()) in SYMPTOM_MAP]
    return max(set(matched), key=matched.count) if matched else "General Condition"

def get_feature_impacts(patient_features=None):
    labels = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol', 'Na-to-K Ratio', 'Condition Factor']
    if not rf_model:
        return {l: 16.6 for l in labels}
    
    global_importances = rf_model.feature_importances_
    
    if patient_features is not None:
        # Calculate Local Impact: Importance * Absolute Magnitude of the Feature
        feat_values = np.abs(patient_features[0])
        # Avoid zero-multiplication for encoded categories
        feat_values = np.array([max(v, 0.1) for v in feat_values])
        
        raw_local = global_importances * feat_values
        norm_local = (raw_local / raw_local.sum()) * 100
        return {labels[i]: float(round(norm_local[i], 2)) for i in range(len(labels))}
    
    return {labels[i]: float(round(global_importances[i] * 100, 2)) for i in range(len(labels))}

# ----------------------------
# Forms & Routes
# ----------------------------
class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Signup')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class TextForm(FlaskForm):
    patient_name = StringField('Patient Name', validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired()])
    sex = SelectField('Gender', choices=[('F', 'Female'), ('M', 'Male')], validators=[DataRequired()])
    bp = SelectField('Blood Pressure', choices=[('HIGH', 'High'), ('NORMAL', 'Normal'), ('LOW', 'Low')], validators=[DataRequired()])
    cholesterol = SelectField('Cholesterol', choices=[('HIGH', 'High'), ('NORMAL', 'Normal')], validators=[DataRequired()])
    na_to_k = FloatField('Na to K Ratio', validators=[DataRequired()])
    symptoms = SelectMultipleField('Symptoms', choices=[
        ('fever', 'Fever'), ('cough', 'Cough'), ('fatigue', 'Fatigue'), ('low_mood', 'Low Mood'),
        ('insomnia', 'Insomnia'), ('acne', 'Acne'), ('chronic_pain', 'Chronic Pain'),
        ('frequent_urination', 'Frequent Urination'), ('short_breath', 'Shortness of Breath'),
        ('sadness', 'Sadness'), ('anxiety', 'Anxiety'), ('nausea', 'Nausea'), ('headache', 'Headache'),
        ('chest_pain', 'Chest Pain'), ('dizziness', 'Dizziness'), ('vomiting', 'Vomiting'),
        ('rash', 'Skin Rash'), ('sore_throat', 'Sore Throat'), ('joint_stiffness', 'Joint Stiffness'),
        ('blurred_vision', 'Blurred Vision'), ('weight_loss', 'Weight Loss'), ('night_sweats', 'Night Sweats'),
        ('palpitations', 'Heart Palpitations'), ('constipation', 'Constipation'), ('diarrhea', 'Diarrhea'),
        ('abdominal_pain', 'Abdominal Pain'), ('muscle_weakness', 'Muscle Weakness'), ('tremor', 'Tremors'),
        ('numbness', 'Numbness'), ('back_pain', 'Back Pain'), ('sneeze', 'Sneezing')
    ], validators=[DataRequired()])
    submit = SubmitField('Analyze')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash("Username already exists", "error")
        else:
            hashed_pw = generate_password_hash(form.password.data)
            db.session.add(User(username=form.username.data, email=form.email.data, password_hash=hashed_pw))
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            is_valid = check_password_hash(user.password_hash, form.password.data)
            print(f"DEBUG: Login attempt for {user.username} - Pass match: {is_valid}")
            if is_valid:
                session['username'] = user.username
                session['user_id'] = user.id
                return redirect(url_for('home'))
        
        print(f"DEBUG: Login failed for username: {form.username.data}")
        flash("Invalid credentials", "error")
    return render_template('login.html', form=form)

@app.route('/index1', methods=['GET', 'POST'])
def index1():
    if 'username' not in session:
        return redirect(url_for('login'))
    form = TextForm()
    if form.validate_on_submit():
        condition = infer_condition(form.symptoms.data)
        sex_enc = sex_map.get(form.sex.data, 0)
        bp_enc = bp_map.get(form.bp.data, 1)
        chol_enc = chol_map.get(form.cholesterol.data, 0)
        try:
            reason_enc = le_reason.transform([condition])[0]
        except Exception:
            reason_enc = 0
        
        nums_scaled = scaler.transform([[form.age.data, form.na_to_k.data]])
        features = np.array([[nums_scaled[0][0], sex_enc, bp_enc, chol_enc, nums_scaled[0][1], reason_enc]])
        
        rf_probs = get_aligned_probs(rf_model, features, le_drug.classes_)
        dt_probs = get_aligned_probs(dt_model, features, le_drug.classes_)
        knn_probs = get_aligned_probs(knn_model, features, le_drug.classes_)
        
        final_probs = (rf_probs * 0.6) + (dt_probs * 0.2) + (knn_probs * 0.2)
        winning_idx = np.argmax(final_probs)
        consensus_drug = le_drug.classes_[winning_idx]
        
        # Calculate raw consensus metrics
        winning_prob = final_probs[winning_idx]
        confidence = round(float(winning_prob * 100), 1)
        
        # Get individual predictions for consistency check
        rf_pred = le_drug.classes_[np.argmax(rf_probs)] if rf_model else "Offline"
        dt_pred = le_drug.classes_[np.argmax(dt_probs)]
        knn_pred = le_drug.classes_[np.argmax(knn_probs)]
        
        unique_preds = len(set([rf_pred, dt_pred, knn_pred]))
        votes = 4 - unique_preds  # Proxy for consensus strength
        
        results = {
            'Random Forest': rf_pred,
            'Decision Tree': dt_pred,
            'KNN': knn_pred
        }
        
        # Get regimen and apply Pediatric Safety Layer
        base_regimen = REGIMEN_DATA.get(condition, REGIMEN_DATA['General Condition'])
        regimen = base_regimen.copy()
        
        is_pediatric = form.age.data < 12
        has_red_flags = any(s in ['chest_pain', 'short_breath', 'blurred_vision'] for s in form.symptoms.data)
        
        if is_pediatric:
            if condition == 'Viral Infection':
                regimen['dosage'] = '160mg - 240mg (Weight Dependent)'
                regimen['notes'] = 'Dose adjusted for Pediatric Safety (15mg/kg standard).'
            else:
                regimen['dosage'] = 'Reduced Pediatric Dose'
            
        if has_red_flags and is_pediatric:
            is_validated = "URGENT REVIEW REQUIRED"
            confidence = min(confidence, 45.0) # Force low confidence for high-risk pediatric cases
        elif (confidence > 65 and unique_preds < 3):
            is_validated = "Validated"
        else:
            is_validated = "Pending Review"

        new_pred = Prediction(
            user_id=session.get('user_id'),
            patient_name=form.patient_name.data,
            age=form.age.data,
            sex=form.sex.data,
            bp=form.bp.data,
            cholesterol=form.cholesterol.data,
            na_to_k=form.na_to_k.data,
            condition=condition,
            predicted_drug=consensus_drug,
            confidence=confidence
        )
        db.session.add(new_pred)
        db.session.commit()

        future_score = 10
        if form.bp.data == 'HIGH':
            future_score += 40
        future_risks = {
            'Cardiovascular': min(future_score + 30, 98),
            'Metabolic': min(future_score + 10, 85),
            'Renal': min(future_score + 20, 90),
            'Neurological': min(future_score, 75)
        }

        return render_template(
            'result.html',
            condition=condition,
            results=results,
            confidence=confidence,
            consensus_drug=consensus_drug,
            is_validated=is_validated,
            votes=votes,
            total_models=3,
            patient={
                'name': form.patient_name.data,
                'age': form.age.data,
                'sex': form.sex.data,
                'bp': form.bp.data,
                'cholesterol': form.cholesterol.data,
                'na_to_k': form.na_to_k.data
            },
            prediction_id=new_pred.id,
            impacts=get_feature_impacts(features),
            regimen=regimen,
            is_pediatric=is_pediatric,
            has_red_flags=has_red_flags,
            model_profiles=MODEL_PROFILES,
            future_risks=future_risks,
            preventative_plan=["Pediatric specialist consultation", "Vital monitoring", "Emergency protocol awareness"] if is_pediatric else ["DASH Diet protocol", "Daily vitals monitoring", "Increased hydration"]
        )
    return render_template('index1.html', form=form)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template(
        'history.html',
        history=Prediction.query.filter_by(user_id=session.get('user_id')).order_by(Prediction.timestamp.desc()).all()
    )

@app.route('/export_pdf/<int:prediction_id>')
def export_pdf(prediction_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    pred = Prediction.query.get_or_404(prediction_id)
    
    # Ownership verification (IDOR Protection)
    if pred.user_id != session.get('user_id'):
        flash("Unauthorized access", "error")
        return redirect(url_for('history'))
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="MediAI Clinical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient: {pred.patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Condition: {pred.condition}", ln=True)
    pdf.cell(200, 10, txt=f"Ensemble Consensus: {pred.predicted_drug}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {pred.confidence}%", ln=True)
    
    response = make_response(pdf.output(dest='S'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Report_{prediction_id}.pdf'
    return response

@app.route('/interactivedashboard')
def interactivedashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Calculate real-time metrics for each model
    rf_y_pred = rf_model.predict(X_test)
    dt_y_pred = dt_model.predict(X_test)
    knn_y_pred = knn_model.predict(X_test)
    
    metrics = {
        'Random Forest': {
            'accuracy': round(accuracy_score(y_test, rf_y_pred) * 100, 1),
            'precision': round(precision_score(y_test, rf_y_pred, average='weighted') * 100, 1),
            'recall': round(recall_score(y_test, rf_y_pred, average='weighted') * 100, 1),
            'f1': round(f1_score(y_test, rf_y_pred, average='weighted') * 100, 1)
        },
        'Decision Tree': {
            'accuracy': round(accuracy_score(y_test, dt_y_pred) * 100, 1),
            'precision': round(precision_score(y_test, dt_y_pred, average='weighted') * 100, 1),
            'recall': round(recall_score(y_test, dt_y_pred, average='weighted') * 100, 1),
            'f1': round(f1_score(y_test, dt_y_pred, average='weighted') * 100, 1)
        },
        'KNN': {
            'accuracy': round(accuracy_score(y_test, knn_y_pred) * 100, 1),
            'precision': round(precision_score(y_test, knn_y_pred, average='weighted') * 100, 1),
            'recall': round(recall_score(y_test, knn_y_pred, average='weighted') * 100, 1),
            'f1': round(f1_score(y_test, knn_y_pred, average='weighted') * 100, 1)
        }
    }
    
    return render_template('interactivedashboard.html', metrics=metrics)

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)