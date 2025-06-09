#!/usr/bin/env python3
"""
CardioInsight AI - Web Application

This is the main application file for the CardioInsight AI web interface.
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import plotly
import json

# Add parent directory to path to import cardioinsight_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Import CardioInsight AI modules
from cardioinsight_ai.main import CardioInsightAI
from web_app.modules.auth import auth_bp
from web_app.modules.data import data_bp
from web_app.modules.analysis import analysis_bp
from web_app.modules.visualization import visualization_bp
from web_app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(data_bp, url_prefix='/data')
app.register_blueprint(analysis_bp, url_prefix='/analysis')
app.register_blueprint(visualization_bp, url_prefix='/visualization')

# Initialize CardioInsight AI system
cardioinsight_system = None

def get_system():
    """
    Get or initialize the CardioInsight AI system.
    
    Returns:
    --------
    system : CardioInsightAI
        CardioInsight AI system.
    """
    global cardioinsight_system
    
    if cardioinsight_system is None:
        logger.info("Initializing CardioInsight AI system...")
        cardioinsight_system = CardioInsightAI()
        
    return cardioinsight_system

@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """
    Render the dashboard page.
    """
    # Get system statistics
    system = get_system()
    
    # Get available models
    ml_models = system.ml_model_manager.get_available_models()
    dl_models = system.dl_model_manager.get_available_models()
    
    # Get recent analyses
    recent_analyses = []
    results_dir = system.config['results_dir']
    
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        for file in result_files[:5]:  # Get 5 most recent
            file_path = os.path.join(results_dir, file)
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    
                recent_analyses.append({
                    'filename': file,
                    'timestamp': result.get('timestamp', ''),
                    'prediction': result.get('prediction', ''),
                    'confidence': result.get('confidence', 0)
                })
            except Exception as e:
                logger.error(f"Error loading result file {file}: {e}")
    
    return render_template(
        'dashboard.html',
        ml_models=ml_models,
        dl_models=dl_models,
        recent_analyses=recent_analyses
    )

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """
    Render the analysis page or process analysis request.
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'ecg_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['ecg_file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # Get analysis parameters
            use_dl = request.form.get('use_dl', 'true') == 'true'
            explain = request.form.get('explain', 'true') == 'true'
            
            try:
                # Get system
                system = get_system()
                
                # Load ECG data
                ecg_data, metadata = system.load_ecg_data(file_path)
                
                # Add sampling rate if provided
                if 'sampling_rate' in request.form and request.form['sampling_rate']:
                    metadata['sampling_rate'] = float(request.form['sampling_rate'])
                
                # Analyze ECG
                results = system.analyze_ecg(ecg_data, metadata, use_dl=use_dl, explain=explain)
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                result_filename = f"analysis_{timestamp}.json"
                result_path = system.save_results(results, result_filename)
                
                # Generate plots
                plots = {}
                
                # ECG plot
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Plot first 2500 samples of first 3 leads
                n_samples = min(2500, ecg_data.shape[0])
                n_leads = min(3, ecg_data.shape[1])
                
                fig = make_subplots(rows=n_leads, cols=1, shared_xaxes=True)
                
                for i in range(n_leads):
                    fig.add_trace(
                        go.Scatter(
                            y=ecg_data[:n_samples, i],
                            name=f"Lead {i+1}"
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    title="ECG Data",
                    height=600,
                    width=1000
                )
                
                plots['ecg'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                # Explanation plot if available
                if explain and results.get('explanation') is not None:
                    # This is a simplified version, in a real app we would create
                    # a proper visualization of the explanation
                    explanation_fig = go.Figure()
                    
                    # Add a heatmap-like visualization
                    # In a real app, we would use the actual explanation data
                    explanation_fig.add_trace(
                        go.Heatmap(
                            z=np.random.rand(10, 10),  # Placeholder
                            colorscale='Viridis'
                        )
                    )
                    
                    explanation_fig.update_layout(
                        title="Explanation Visualization",
                        height=400,
                        width=600
                    )
                    
                    plots['explanation'] = json.dumps(explanation_fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'analysis_results.html',
                    results=results,
                    plots=plots,
                    filename=filename
                )
                
            except Exception as e:
                logger.error(f"Error analyzing ECG: {e}")
                flash(f"Error analyzing ECG: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('analyze.html')

@app.route('/multimodal', methods=['GET', 'POST'])
def multimodal():
    """
    Render the multimodal analysis page or process multimodal analysis request.
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'ecg_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['ecg_file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            try:
                # Get system
                system = get_system()
                
                # Load ECG data
                ecg_data, metadata = system.load_ecg_data(file_path)
                
                # Add sampling rate if provided
                if 'sampling_rate' in request.form and request.form['sampling_rate']:
                    metadata['sampling_rate'] = float(request.form['sampling_rate'])
                
                # Get clinical data from form
                clinical_data = {
                    "demographics": {
                        "age": int(request.form.get('age', 0)),
                        "gender": request.form.get('gender', ''),
                        "height": float(request.form.get('height', 0)),
                        "weight": float(request.form.get('weight', 0)),
                        "bmi": float(request.form.get('bmi', 0))
                    },
                    "symptoms": request.form.get('symptoms', '').split(','),
                    "medical_history": request.form.get('medical_history', '').split(','),
                    "medications": request.form.get('medications', '').split(','),
                    "lab_results": {},
                    "vital_signs": {
                        "heart_rate": int(request.form.get('heart_rate', 0)),
                        "blood_pressure": request.form.get('blood_pressure', ''),
                        "respiratory_rate": int(request.form.get('respiratory_rate', 0)),
                        "oxygen_saturation": int(request.form.get('oxygen_saturation', 0))
                    }
                }
                
                # Analyze with multimodal fusion
                results = system.analyze_with_multimodal(ecg_data, clinical_data, metadata)
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                result_filename = f"multimodal_analysis_{timestamp}.json"
                result_path = system.save_results(results, result_filename)
                
                # Generate plots (similar to analyze route)
                plots = {}
                
                # ECG plot
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Plot first 2500 samples of first 3 leads
                n_samples = min(2500, ecg_data.shape[0])
                n_leads = min(3, ecg_data.shape[1])
                
                fig = make_subplots(rows=n_leads, cols=1, shared_xaxes=True)
                
                for i in range(n_leads):
                    fig.add_trace(
                        go.Scatter(
                            y=ecg_data[:n_samples, i],
                            name=f"Lead {i+1}"
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    title="ECG Data",
                    height=600,
                    width=1000
                )
                
                plots['ecg'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'multimodal_results.html',
                    results=results,
                    plots=plots,
                    filename=filename,
                    clinical_data=clinical_data
                )
                
            except Exception as e:
                logger.error(f"Error in multimodal analysis: {e}")
                flash(f"Error in multimodal analysis: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('multimodal.html')

@app.route('/teaching', methods=['GET'])
def teaching():
    """
    Render the teaching module page.
    """
    # Get system
    system = get_system()
    
    # Get available quizzes
    quizzes = []
    
    for difficulty in ['easy', 'medium', 'hard']:
        for n_questions in [5, 10, 15]:
            quizzes.append({
                'id': f"{difficulty}_{n_questions}",
                'title': f"{difficulty.capitalize()} Quiz ({n_questions} questions)",
                'difficulty': difficulty,
                'n_questions': n_questions
            })
    
    return render_template('teaching.html', quizzes=quizzes)

@app.route('/teaching/quiz/<quiz_id>', methods=['GET'])
def start_quiz(quiz_id):
    """
    Start a quiz.
    """
    try:
        # Parse quiz ID
        difficulty, n_questions = quiz_id.split('_')
        n_questions = int(n_questions)
        
        # Get system
        system = get_system()
        
        # Create quiz
        quiz = system.create_teaching_quiz(n_questions=n_questions, difficulty=difficulty)
        
        # Store quiz in session
        session['quiz'] = quiz
        session['current_question'] = 0
        session['answers'] = []
        
        return redirect(url_for('quiz_question'))
        
    except Exception as e:
        logger.error(f"Error starting quiz: {e}")
        flash(f"Error starting quiz: {str(e)}", 'error')
        return redirect(url_for('teaching'))

@app.route('/teaching/quiz/question', methods=['GET', 'POST'])
def quiz_question():
    """
    Display a quiz question or process answer.
    """
    # Check if quiz exists in session
    if 'quiz' not in session:
        flash('No active quiz', 'error')
        return redirect(url_for('teaching'))
    
    quiz = session['quiz']
    current_question = session['current_question']
    
    if current_question >= len(quiz['questions']):
        # Quiz completed
        return redirect(url_for('quiz_results'))
    
    question = quiz['questions'][current_question]
    
    if request.method == 'POST':
        # Process answer
        selected_option = request.form.get('answer')
        
        if selected_option:
            # Check answer
            is_correct = selected_option == question['correct_answer']
            
            # Store answer
            session['answers'].append({
                'question_id': question['id'],
                'selected_answer': selected_option,
                'correct_answer': question['correct_answer'],
                'is_correct': is_correct
            })
            
            # Move to next question
            session['current_question'] = current_question + 1
            
            return redirect(url_for('quiz_question'))
    
    return render_template('quiz_question.html', question=question)

@app.route('/teaching/quiz/results', methods=['GET'])
def quiz_results():
    """
    Display quiz results.
    """
    # Check if quiz exists in session
    if 'quiz' not in session or 'answers' not in session:
        flash('No active quiz', 'error')
        return redirect(url_for('teaching'))
    
    quiz = session['quiz']
    answers = session['answers']
    
    # Calculate score
    correct_answers = sum(1 for answer in answers if answer['is_correct'])
    total_questions = len(quiz['questions'])
    score = correct_answers / total_questions if total_questions > 0 else 0
    
    return render_template(
        'quiz_results.html',
        quiz=quiz,
        answers=answers,
        score=score,
        correct_answers=correct_answers,
        total_questions=total_questions
    )

@app.route('/remote', methods=['GET', 'POST'])
def remote():
    """
    Render the remote healthcare page or process remote healthcare request.
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'ecg_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['ecg_file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            try:
                # Get system
                system = get_system()
                
                # Load ECG data
                ecg_data, metadata = system.load_ecg_data(file_path)
                
                # Add metadata from form
                metadata['patient_id'] = request.form.get('patient_id', '')
                metadata['location'] = request.form.get('location', '')
                metadata['requires_consultation'] = request.form.get('requires_consultation', 'false') == 'true'
                metadata['priority'] = request.form.get('priority', 'normal')
                
                # Process for remote consultation
                result = system.process_for_remote(ecg_data, metadata)
                
                return render_template(
                    'remote_results.html',
                    result=result,
                    filename=filename,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Error in remote healthcare: {e}")
                flash(f"Error in remote healthcare: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('remote.html')

@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('about.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Shutdown the CardioInsight AI system and the web application.
    """
    global cardioinsight_system
    
    if cardioinsight_system is not None:
        logger.info("Shutting down CardioInsight AI system...")
        cardioinsight_system.shutdown()
        cardioinsight_system = None
    
    return jsonify({'status': 'success'})

@app.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors.
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors.
    """
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)

