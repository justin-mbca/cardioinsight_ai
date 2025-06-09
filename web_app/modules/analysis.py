"""
CardioInsight AI - Analysis Module

This module handles ECG analysis for the CardioInsight AI web application.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from web_app.modules.auth import login_required, role_required

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/')
@login_required
def index():
    """
    Display the analysis dashboard.
    """
    # Get CardioInsight AI system
    from app import get_system
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
        
        for file in result_files[:10]:  # Get 10 most recent
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
        'analysis/index.html',
        ml_models=ml_models,
        dl_models=dl_models,
        recent_analyses=recent_analyses
    )

@analysis_bp.route('/basic', methods=['GET', 'POST'])
#@login_required
# @login_required  # <-- comment this out or delete
def basic():
    """
    Perform basic ECG analysis.
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
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # Get analysis parameters
            use_dl = request.form.get('use_dl', 'true') == 'true'
            explain = request.form.get('explain', 'true') == 'true'
            
            try:
                # Get CardioInsight AI system
                from app import get_system
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
                    'analysis/results.html',
                    results=results,
                    plots=plots,
                    filename=filename
                )
                
            except Exception as e:
                logger.error(f"Error analyzing ECG: {e}")
                flash(f"Error analyzing ECG: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('analysis/basic.html')

@analysis_bp.route('/multimodal', methods=['GET', 'POST'])
@login_required
def multimodal():
    """
    Perform multimodal ECG analysis.
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
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            try:
                # Get CardioInsight AI system
                from app import get_system
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
                
                # Generate plots (similar to basic analysis)
                plots = {}
                
                # ECG plot
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
                    'analysis/multimodal_results.html',
                    results=results,
                    plots=plots,
                    filename=filename,
                    clinical_data=clinical_data
                )
                
            except Exception as e:
                logger.error(f"Error in multimodal analysis: {e}")
                flash(f"Error in multimodal analysis: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('analysis/multimodal.html')

@analysis_bp.route('/dynamic', methods=['GET', 'POST'])
@login_required
def dynamic():
    """
    Perform dynamic ECG analysis (Holter data).
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
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            try:
                # Get CardioInsight AI system
                from app import get_system
                system = get_system()
                
                # Load ECG data
                ecg_data, metadata = system.load_ecg_data(file_path)
                
                # Add sampling rate if provided
                if 'sampling_rate' in request.form and request.form['sampling_rate']:
                    metadata['sampling_rate'] = float(request.form['sampling_rate'])
                
                # Get analysis parameters
                window_size = int(request.form.get('window_size', 3600))  # Default: 1 hour
                step_size = int(request.form.get('step_size', 1800))  # Default: 30 minutes
                
                # Analyze dynamic ECG
                results = system.analyze_dynamic_ecg(ecg_data, metadata, window_size=window_size, step_size=step_size)
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                result_filename = f"dynamic_analysis_{timestamp}.json"
                result_path = system.save_results(results, result_filename)
                
                # Generate plots
                plots = {}
                
                # Overview plot
                fig = go.Figure()
                
                # Add events
                if 'events' in results:
                    for event in results['events']:
                        fig.add_trace(
                            go.Scatter(
                                x=[event['time'], event['time']],
                                y=[0, 1],
                                mode='lines',
                                name=event['type'],
                                line=dict(color='red', width=2)
                            )
                        )
                
                fig.update_layout(
                    title="Dynamic ECG Analysis - Events",
                    height=400,
                    width=1000,
                    xaxis_title="Time (s)",
                    yaxis_title="Event"
                )
                
                plots['events'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'analysis/dynamic_results.html',
                    results=results,
                    plots=plots,
                    filename=filename
                )
                
            except Exception as e:
                logger.error(f"Error in dynamic ECG analysis: {e}")
                flash(f"Error in dynamic ECG analysis: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('analysis/dynamic.html')

@analysis_bp.route('/results/<result_id>')
@login_required
def view_result(result_id):
    """
    View a specific analysis result.
    """
    # Get CardioInsight AI system
    from app import get_system
    system = get_system()
    
    # Get result file path
    results_dir = system.config['results_dir']
    result_path = os.path.join(results_dir, result_id)
    
    if not os.path.exists(result_path):
        flash(f'Result {result_id} not found', 'error')
        return redirect(url_for('analysis.index'))
    
    try:
        # Load result
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        # Generate plots based on result type
        plots = {}
        
        # For now, just return the result
        return render_template(
            'analysis/view_result.html',
            result=result,
            plots=plots,
            result_id=result_id
        )
        
    except Exception as e:
        logger.error(f"Error viewing result {result_id}: {e}")
        flash(f"Error viewing result: {str(e)}", 'error')
        return redirect(url_for('analysis.index'))

@analysis_bp.route('/batch', methods=['GET', 'POST'])
@login_required
@role_required('researcher')
def batch():
    """
    Perform batch analysis on multiple files.
    """
    if request.method == 'POST':
        # Get selected files
        file_ids = request.form.getlist('file_ids')
        
        if not file_ids:
            flash('No files selected', 'error')
            return redirect(request.url)
        
        # Get analysis parameters
        use_dl = request.form.get('use_dl', 'true') == 'true'
        explain = request.form.get('explain', 'true') == 'true'
        
        try:
            # Get CardioInsight AI system
            from app import get_system
            system = get_system()
            
            # Process each file
            results = []
            
            for file_id in file_ids:
                # Get file path
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads', file_id)
                
                if not os.path.exists(file_path):
                    logger.warning(f"File {file_id} not found, skipping")
                    continue
                
                # Load ECG data
                ecg_data, metadata = system.load_ecg_data(file_path)
                
                # Analyze ECG
                result = system.analyze_ecg(ecg_data, metadata, use_dl=use_dl, explain=explain)
                
                # Add file info
                result['file_id'] = file_id
                
                # Save result
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                result_filename = f"analysis_{file_id}_{timestamp}.json"
                result_path = system.save_results(result, result_filename)
                
                # Add to results
                results.append(result)
            
            # Save batch results
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            batch_result_filename = f"batch_analysis_{timestamp}.json"
            batch_result_path = os.path.join(system.config['results_dir'], batch_result_filename)
            
            with open(batch_result_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'n_files': len(file_ids),
                    'n_processed': len(results),
                    'results': results
                }, f)
            
            flash(f'Batch analysis completed. Processed {len(results)} of {len(file_ids)} files.', 'success')
            return redirect(url_for('analysis.index'))
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            flash(f"Error in batch analysis: {str(e)}", 'error')
            return redirect(request.url)
    
    # Get list of uploaded files
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    files = []
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            files.append({
                'name': filename,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'path': file_path
            })
    
    # Sort files by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return render_template('analysis/batch.html', files=files)

