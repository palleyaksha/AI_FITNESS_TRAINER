from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import uuid
import base64
import json
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import sqlalchemy as sa
import io
import traceback

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fitness.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Gemini AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Define models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    login_streak = db.Column(db.Integer, default=0)
    last_login = db.Column(db.DateTime, default=None)
    total_workout_hours = db.Column(db.Float, default=0.0)
    goal_calories = db.Column(db.Integer, default=1000)
    badges = db.relationship('Badge', backref='user', lazy=True)
    workouts = db.relationship('Workout', backref='user', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    image = db.Column(db.String(100), nullable=False)
    requirement_hours = db.Column(db.Float, nullable=False)
    date_earned = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Workout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exercise_type = db.Column(db.String(50), nullable=False)
    duration = db.Column(db.Integer, nullable=False)  # in seconds
    calories = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    video_path = db.Column(db.String(200), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Define the model architecture for exercise detection
class BEiTLSTMModel(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=2, dropout=0.2):
        super(BEiTLSTMModel, self).__init__()
        # Load pre-trained BEiT model
        self.beit = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Get the feature dimension from BEiT
        self.feature_dim = self.beit.num_features
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Extract features using BEiT
        batch_size = x.size(0)
        features = self.beit(x)  # [batch_size, feature_dim]
        
        # Reshape for LSTM: [batch_size, sequence_length=1, feature_dim]
        features = features.unsqueeze(1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Classification layer
        output = self.fc(lstm_out)
        
        return output

# Global variables
class_names = ["jumping_jacks", "push_ups", "rotating_toe_touches", "squats"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Badge definitions
BADGES = [
    {"name": "Beginner", "description": "Completed 1 hour of workouts", "image": "beginner.png", "requirement_hours": 1.0},
    {"name": "Enthusiast", "description": "Completed 5 hours of workouts", "image": "enthusiast.png", "requirement_hours": 5.0},
    {"name": "Dedicated", "description": "Completed 10 hours of workouts", "image": "dedicated.png", "requirement_hours": 10.0},
    {"name": "Athlete", "description": "Completed 20 hours of workouts", "image": "athlete.png", "requirement_hours": 20.0},
    {"name": "Champion", "description": "Completed 30 hours of workouts", "image": "champion.png", "requirement_hours": 30.0},
    {"name": "Master", "description": "Completed 50 hours of workouts", "image": "master.png", "requirement_hours": 50.0},
    {"name": "Elite", "description": "Completed 75 hours of workouts", "image": "elite.png", "requirement_hours": 75.0},
    {"name": "Legend", "description": "Completed 100 hours of workouts", "image": "legend.png", "requirement_hours": 100.0},
    {"name": "Superhuman", "description": "Completed 150 hours of workouts", "image": "superhuman.png", "requirement_hours": 150.0},
    {"name": "Immortal", "description": "Completed 200 hours of workouts", "image": "immortal.png", "requirement_hours": 200.0}
]

# Load the model
def load_model():
    global model
    model = BEiTLSTMModel(num_classes=len(class_names))
    
    # Look for model file in the current directory
    model_files = [f for f in os.listdir('.') if f.endswith('.pt') or f.endswith('.pth')]
    
    if model_files:
        model_path = model_files[0]
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize with random weights if model loading fails
            print("Initializing model with random weights")
    else:
        print("No model file found. Initializing model with random weights")
    
    model.to(device)
    model.eval()

# Preprocess frame for model input
def preprocess_frame(frame):
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transformations
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Preprocess the image
        tensor = preprocess(pil_image).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

# Process a single frame for exercise detection
def process_frame(frame):
    with torch.no_grad():
        # Preprocess the frame
        input_tensor = preprocess_frame(frame)
        if input_tensor is None:
            return None, None
        
        # Move tensor to device
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        try:
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = class_names[predicted_idx.item()]
            confidence_value = confidence.item()
            
            return predicted_class, confidence_value
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None

# Check and award badges based on total workout hours
def check_and_award_badges(user_id):
    user = User.query.get(user_id)
    if not user:
        return
    
    # Get user's current badges
    user_badges = Badge.query.filter_by(user_id=user_id).all()
    user_badge_names = [badge.name for badge in user_badges]
    
    # Check for new badges
    for badge_info in BADGES:
        if badge_info["name"] not in user_badge_names and user.total_workout_hours >= badge_info["requirement_hours"]:
            new_badge = Badge(
                name=badge_info["name"],
                description=badge_info["description"],
                image=badge_info["image"],
                requirement_hours=badge_info["requirement_hours"],
                user_id=user_id
            )
            db.session.add(new_badge)
            db.session.commit()
            flash(f'Congratulations! You earned the {badge_info["name"]} badge!', 'success')

# AI Assistant function
def get_ai_response(user_message, chat_history=None):
    try:
        # Configure Gemini
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Define the model
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40
            }
        )
        
        # Prepare fitness assistant instructions
        fitness_instructions = """You are an AI Fitness Assistant. Your purpose is to provide helpful information and advice ONLY about:
        - Fitness and exercise routines
        - Workout techniques and form
        - Diet and nutrition for fitness goals
        - Health and wellness tips
        - Fitness motivation and planning
        - Exercise science and physiology
        
        please follow the instructions properly and give the output carefully and not other than mentioned topic below.
        If asked about topics outside of fitness, health, diet, and exercise, politely redirect the conversation 
        back to fitness-related topics. For example: "I'm your fitness assistant, so I can help you with workout plans, 
        diet advice, or exercise techniques. How can I assist you with your fitness journey today?"
        
        Be friendly, encouraging, and provide scientifically accurate information. When giving advice, consider 
        safety first and recommend consulting healthcare professionals for personalized medical advice."""
        
        # Prepare the conversation history
        messages = []
        
        # Add chat history if provided
        if chat_history and len(chat_history) > 0:
            for message in chat_history:
                role = message.get('role', 'user')
                content = message.get('content', '')
                if role and content:
                    # Only add user and assistant messages (no system messages)
                    if role in ['user', 'assistant']:
                        messages.append({"role": role, "parts": [content]})
        
        # If this is the first message, include the instructions with the user message
        if not messages:
            # Combine instructions with the user's first message
            combined_message = f"{fitness_instructions}\n\nUser: {user_message}"
            messages.append({"role": "user", "parts": [combined_message]})
        else:
            # For subsequent messages, just add the user message normally
            messages.append({"role": "user", "parts": [user_message]})
        
        # Generate the response
        response = model.generate_content(messages)
        
        return response.text
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm sorry, I encountered an error. Please try again."

# Custom filter for Jinja templates
@app.template_filter('format_duration')
def format_duration(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        age = request.form['age']
        gender = request.form['gender']
        weight = request.form['weight']
        height = request.form['height']
        password = request.form['password']
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(
            name=name,
            username=username,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            password=hashed_password,
            total_workout_hours=0.0,
            goal_calories=1000  # Default goal calories
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Admin login
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            session['username'] = 'admin'
            session['is_admin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        # Regular user login
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = user.id
            session['is_admin'] = False
            
            # Check login streak
            now = datetime.utcnow()
            if user.last_login:
                # Check if last login was yesterday
                yesterday = now - timedelta(days=1)
                if yesterday.date() == user.last_login.date():
                    user.login_streak += 1
                    
                    # Award badges based on streak
                    if user.login_streak == 3:
                        new_badge = Badge(
                            name="3-Day Streak", 
                            description="Logged in for 3 consecutive days", 
                            image="streak3.png",
                            requirement_hours=0.0,
                            user_id=user.id
                        )
                        db.session.add(new_badge)
                        flash('Congratulations! You earned the 3-Day Streak badge!', 'success')
                    elif user.login_streak == 7:
                        new_badge = Badge(
                            name="Weekly Warrior", 
                            description="Logged in for 7 consecutive days", 
                            image="streak7.png",
                            requirement_hours=0.0,
                            user_id=user.id
                        )
                        db.session.add(new_badge)
                        flash('Congratulations! You earned the Weekly Warrior badge!', 'success')
                    elif user.login_streak == 30:
                        new_badge = Badge(
                            name="Monthly Master", 
                            description="Logged in for 30 consecutive days", 
                            image="streak30.png",
                            requirement_hours=0.0,
                            user_id=user.id
                        )
                        db.session.add(new_badge)
                        flash('Congratulations! You earned the Monthly Master badge!', 'success')
                elif now.date() != user.last_login.date():  # Not consecutive but not same day
                    user.login_streak = 1
            else:
                user.login_streak = 1
            
            # Update last login time
            user.last_login = now
            db.session.commit()
            
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))
    
    if session.get('is_admin', False):
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.filter_by(id=session['user_id']).first()
    badges = Badge.query.filter_by(user_id=session['user_id']).all()
    
    # Get recent workouts
    recent_workouts = Workout.query.filter_by(user_id=session['user_id']).order_by(Workout.date.desc()).limit(5).all()
    
    # Calculate BMI
    bmi = round(user.weight / ((user.height / 100) ** 2), 2)
    
    # Determine BMI category
    bmi_category = ""
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal weight"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    # Calculate total calories burned
    total_calories = db.session.query(db.func.sum(Workout.calories)).filter(Workout.user_id == session['user_id']).scalar() or 0
    
    # Calculate calories burned today
    today = datetime.utcnow().date()
    today_calories = db.session.query(db.func.sum(Workout.calories)).filter(
        Workout.user_id == session['user_id'],
        db.func.date(Workout.date) == today
    ).scalar() or 0
    
    # Calculate total workout time
    total_duration = db.session.query(db.func.sum(Workout.duration)).filter(Workout.user_id == session['user_id']).scalar() or 0
    total_hours = total_duration // 3600
    total_minutes = (total_duration % 3600) // 60
    
    return render_template(
        'dashboard.html',
        user=user,
        badges=badges,
        recent_workouts=recent_workouts,
        bmi=bmi,
        bmi_category=bmi_category,
        total_calories=total_calories,
        today_calories=today_calories,
        total_hours=total_hours,
        total_minutes=total_minutes
    )

@app.route('/admin')
def admin_dashboard():
    if not session.get('logged_in') or not session.get('is_admin', False):
        flash('You do not have permission to access the admin dashboard', 'danger')
        return redirect(url_for('login'))
    
    users = User.query.all()
    
    return render_template('admin.html', users=users, now=datetime.utcnow)

@app.route('/video_detection')
def video_detection():
    if not session.get('logged_in'):
        flash('Please log in to access this feature', 'warning')
        return redirect(url_for('login'))
    
    return render_template('video_detection.html')

@app.route('/live_monitor')
def live_monitor():
    if not session.get('logged_in'):
        flash('Please log in to access this feature', 'warning')
        return redirect(url_for('login'))
    
    return render_template('live_monitor.html')

@app.route('/profile')
def profile():
    if not session.get('logged_in'):
        flash('Please log in to access your profile', 'warning')
        return redirect(url_for('login'))
    
    if session.get('is_admin', False):
        flash('Admin does not have a profile', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.filter_by(id=session['user_id']).first()
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        user = User.query.filter_by(id=session['user_id']).first()
        
        # Update user details
        user.name = request.form['name']
        user.age = request.form['age']
        user.gender = request.form['gender']
        user.weight = request.form['weight']
        user.height = request.form['height']
        
        # Update password if provided
        if request.form['password'] and len(request.form['password']) > 0:
            user.password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('profile'))

@app.route('/update_calorie_goal', methods=['POST'])
def update_calorie_goal():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        goal_calories = request.form.get('goal_calories', type=int)
        
        if not goal_calories or goal_calories <= 0:
            flash('Please enter a valid calorie goal', 'danger')
            return redirect(url_for('profile'))
        
        user = User.query.filter_by(id=session['user_id']).first()
        user.goal_calories = goal_calories
        db.session.commit()
        
        flash('Calorie goal updated successfully!', 'success')
        return redirect(url_for('profile'))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('profile'))

@app.route('/achievements')
def achievements():
    if not session.get('logged_in'):
        flash('Please log in to view your achievements', 'warning')
        return redirect(url_for('login'))
    
    if session.get('is_admin', False):
        flash('Admin does not have achievements', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.filter_by(id=session['user_id']).first()
    badges = Badge.query.filter_by(user_id=session['user_id']).order_by(Badge.date_earned).all()
    
    # Get all possible badges
    all_badges = BADGES.copy()
    
    # Add streak badges
    streak_badges = [
        {"name": "3-Day Streak", "description": "Logged in for 3 consecutive days", "image": "streak3.png", "requirement_hours": 0.0},
        {"name": "Weekly Warrior", "description": "Logged in for 7 consecutive days", "image": "streak7.png", "requirement_hours": 0.0},
        {"name": "Monthly Master", "description": "Logged in for 30 consecutive days", "image": "streak30.png", "requirement_hours": 0.0}
    ]
    all_badges.extend(streak_badges)
    
    # Get earned badge names
    earned_badge_names = [badge.name for badge in badges]
    
    return render_template('achievements.html', user=user, badges=badges, all_badges=all_badges, earned_badge_names=earned_badge_names)

@app.route('/workout_history')
def workout_history():
    if not session.get('logged_in'):
        flash('Please log in to view your workout history', 'warning')
        return redirect(url_for('login'))
    
    if session.get('is_admin', False):
        flash('Admin does not have workout history', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.filter_by(id=session['user_id']).first()
    
    # Get all workouts for the user, ordered by date (newest first)
    workouts = Workout.query.filter_by(user_id=session['user_id']).order_by(Workout.date.desc()).all()
    
    # Calculate total stats
    total_duration = sum(workout.duration for workout in workouts)
    total_calories = sum(workout.calories for workout in workouts)
    
    # Calculate calories burned today
    today = datetime.utcnow().date()
    today_calories = sum(workout.calories for workout in workouts if workout.date.date() == today)
    
    # Group workouts by exercise type for statistics
    exercise_stats = {}
    for workout in workouts:
        exercise_type = workout.exercise_type
        if exercise_type not in exercise_stats:
            exercise_stats[exercise_type] = {
                'count': 0,
                'total_duration': 0,
                'total_calories': 0
            }
        exercise_stats[exercise_type]['count'] += 1
        exercise_stats[exercise_type]['total_duration'] += workout.duration
        exercise_stats[exercise_type]['total_calories'] += workout.calories
    
    return render_template(
        'workout_history.html', 
        user=user, 
        workouts=workouts,
        total_duration=total_duration,
        total_calories=total_calories,
        today_calories=today_calories,
        exercise_stats=exercise_stats
    )

@app.route('/settings')
def settings():
    if not session.get('logged_in'):
        flash('Please log in to access settings', 'warning')
        return redirect(url_for('login'))
    
    if session.get('is_admin', False):
        flash('Admin does not have settings', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.filter_by(id=session['user_id']).first()
    return render_template('settings.html', user=user)

@app.route('/process_video', methods=['POST'])
def process_video():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    try:
        # Create a temporary file to process the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            temp_filepath = temp_file.name
        
        # Process the video
        cap = cv2.VideoCapture(temp_filepath)
        if not cap.isOpened():
            os.unlink(temp_filepath)  # Delete the temporary file
            return jsonify({'error': 'Could not open video file'}), 500
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames
        predictions = []
        processed_frames = []
        frame_count = 0
        
        with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for speed
                if frame_count % 10 != 0:
                    frame_count += 1
                    continue
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(frame_rgb)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                
                # Get prediction
                predicted_class, confidence = process_frame(frame)
                
                if predicted_class:
                    predictions.append(predicted_class)
                    
                    # Add text to frame
                    cv2.putText(
                        frame,
                        f"{predicted_class} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Save processed frame (every 30th processed frame)
                    if len(processed_frames) < 5 and frame_count % 30 == 0:
                        _, buffer = cv2.imencode('.jpg', frame)
                        processed_frames.append(base64.b64encode(buffer).decode('utf-8'))
                
                frame_count += 1
        
        cap.release()
        
        # Delete the temporary file
        os.unlink(temp_filepath)
        
        # Count exercise repetitions (simplified)
        exercise_counts = {}
        for exercise in class_names:
            exercise_counts[exercise] = predictions.count(exercise)
        
        # Determine primary exercise
        primary_exercise = max(exercise_counts, key=exercise_counts.get) if exercise_counts else None
        
        # Calculate estimated calories (simplified)
        duration_seconds = total_frames / fps
        calories_per_minute = {
            "jumping_jacks": 8,
            "push_ups": 7,
            "rotating_toe_touches": 6,
            "squats": 8
        }
        
        estimated_calories = 0
        if primary_exercise:
            estimated_calories = int((calories_per_minute.get(primary_exercise, 5) * duration_seconds) / 60)
        
        # Save workout to database
        if primary_exercise and session.get('user_id'):
            # Use current time for the workout date
            current_time = datetime.utcnow()
            
            # Generate a unique identifier for the workout (not an actual file path)
            workout_id = f"video_{session.get('username')}_{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
            
            new_workout = Workout(
                exercise_type=primary_exercise,
                duration=int(duration_seconds),
                calories=estimated_calories,
                video_path=workout_id,  # Store identifier instead of file path
                user_id=session['user_id'],
                date=current_time
            )
            db.session.add(new_workout)
            
            # Update user's total workout hours
            user = User.query.get(session['user_id'])
            user.total_workout_hours += duration_seconds / 3600
            db.session.commit()
            
            # Check for new badges
            check_and_award_badges(session['user_id'])
        
        return jsonify({
            'success': True,
            'primary_exercise': primary_exercise,
            'exercise_counts': exercise_counts,
            'duration': int(duration_seconds),
            'calories': estimated_calories,
            'processed_frames': processed_frames
        })
    
    except Exception as e:
        # Clean up on error
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.unlink(temp_filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_webcam_frame():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Get the base64 image from the request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decode the base64 image
        try:
            encoded_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to decode image data'}), 400
        except Exception as e:
            print(f"Error decoding image: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Error decoding image: {str(e)}'}), 400
        
        # Process with MediaPipe
        with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as pose:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(frame_rgb)
            
            # Draw pose landmarks
            pose_detected = False
            pose_landmarks = []
            
            if results.pose_landmarks:
                pose_detected = True
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Extract landmarks for frontend visualization
                for landmark in results.pose_landmarks.landmark:
                    pose_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
        
        # Get prediction if pose was detected
        predicted_class = None
        confidence = 0
        
        if pose_detected:
            predicted_class, confidence = process_frame(frame)
        
        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': processed_image,
            'pose_detected': pose_detected,
            'pose_landmarks': pose_landmarks,
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error processing frame: {str(e)}'}), 500

@app.route('/save_workout', methods=['POST'])
def save_workout():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        exercise_type = data.get('exercise_type')
        duration = data.get('duration')
        calories = data.get('calories')
        
        if not all([exercise_type, duration, calories]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Use current time for the workout date
        current_time = datetime.utcnow()
        
        # Create a unique identifier for live workouts (not an actual file path)
        workout_id = f"live_{session.get('username')}_{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
        
        new_workout = Workout(
            exercise_type=exercise_type,
            duration=duration,
            calories=calories,
            video_path=workout_id,  # Store identifier instead of file path
            user_id=session['user_id'],
            date=current_time
        )
        
        db.session.add(new_workout)
        
        # Update user's total workout hours
        user = User.query.get(session['user_id'])
        user.total_workout_hours += duration / 3600
        db.session.commit()
        
        # Check for new badges
        check_and_award_badges(session['user_id'])
        
        return jsonify({'success': True, 'workout_id': new_workout.id})
    
    except Exception as e:
        db.session.rollback()
        print(f"Error saving workout: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ai_assistant', methods=['POST'])
def ai_assistant():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        # Get AI response
        ai_response = get_ai_response(user_message, chat_history)
        
        # Save chat history to database if user is logged in
        if session.get('user_id'):
            # Save user message
            user_chat = ChatHistory(
                role='user',
                content=user_message,
                user_id=session['user_id'],
                timestamp=datetime.utcnow()
            )
            db.session.add(user_chat)
            
            # Save AI response
            ai_chat = ChatHistory(
                role='assistant',
                content=ai_response,
                user_id=session['user_id'],
                timestamp=datetime.utcnow()
            )
            db.session.add(ai_chat)
            db.session.commit()
        
        return jsonify({'success': True, 'response': ai_response})
    
    except Exception as e:
        print(f"Error in AI assistant: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Create a function to initialize the database and model
def initialize_app():
    with app.app_context():
        # First create all tables
        db.create_all()
        
        # Now check if we need to update existing users with goal_calories
        try:
            # Check if the table exists before trying to query it
            if db.engine.dialect.has_table(db.engine, 'user'):
                # Check if goal_calories column exists
                inspector = db.inspect(db.engine)
                columns = [column['name'] for column in inspector.get_columns('user')]
                
                # If goal_calories wasn't in the columns, it's a new addition
                if 'goal_calories' not in columns:
                    # We need to add the column
                    with db.engine.connect() as conn:
                        conn.execute(sa.text("ALTER TABLE user ADD COLUMN goal_calories INTEGER DEFAULT 1000"))
                        conn.commit()
                    print("Added goal_calories column to user table")
                
                # Set default goal_calories for existing users
                users = User.query.all()
                for user in users:
                    if not hasattr(user, 'goal_calories') or user.goal_calories is None:
                        user.goal_calories = 1000
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error updating database schema: {e}")
        
        load_model()
        
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin_password = generate_password_hash('admin', method='pbkdf2:sha256')
            admin = User(
                name='Admin',
                username='admin',
                age=30,
                gender='Other',
                weight=70,
                height=175,
                password=admin_password,
                total_workout_hours=0.0,
                goal_calories=1000
            )
            db.session.add(admin)
            db.session.commit()

# Add Jinja2 context processor to make datetime available in templates
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}

if __name__ == '__main__':
    # Initialize the app before running
    initialize_app()
    app.run(debug=True)
