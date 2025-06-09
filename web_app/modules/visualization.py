"""
CardioInsight AI - Visualization Module

This module handles visualization for the CardioInsight AI web application.
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
visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/')
@login_required
def index():
    """
    Display the visualization dashboard.
    """
    return render_template('visualization/index.html')

@visualization_bp.route('/ecg', methods=['GET', 'POST'])
@login_required
def ecg():
    """
    Visualize ECG data.
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
                
                # Get visualization parameters
                plot_type = request.form.get('plot_type', 'single')
                n_seconds = float(request.form.get('n_seconds', 10))
                
                # Calculate number of samples to show
                sampling_rate = metadata.get('sampling_rate', 250)
                n_samples = int(n_seconds * sampling_rate)
                
                # Generate plots
                plots = {}
                
                if plot_type == 'single':
                    # Single lead plot
                    lead_idx = int(request.form.get('lead_idx', 0))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            y=ecg_data[:n_samples, lead_idx],
                            name=f"Lead {lead_idx+1}"
                        )
                    )
                    
                    fig.update_layout(
                        title=f"ECG Data - Lead {lead_idx+1}",
                        height=400,
                        width=1000,
                        xaxis_title="Sample",
                        yaxis_title="Amplitude"
                    )
                    
                    plots['single'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
                elif plot_type == 'multi':
                    # Multi-lead plot
                    n_leads = min(12, ecg_data.shape[1])
                    
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
                        title="ECG Data - Multiple Leads",
                        height=100 * n_leads,
                        width=1000,
                        xaxis_title="Sample",
                        yaxis_title="Amplitude"
                    )
                    
                    plots['multi'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
                elif plot_type == '12lead':
                    # 12-lead plot (3x4 grid)
                    if ecg_data.shape[1] < 12:
                        flash('ECG data has fewer than 12 leads', 'warning')
                        return redirect(request.url)
                    
                    # Create 3x4 grid
                    fig = make_subplots(
                        rows=3, cols=4,
                        subplot_titles=[f"Lead {i+1}" for i in range(12)]
                    )
                    
                    # Add traces
                    lead_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # I, II, III, aVR, aVL, aVF, V1-V6
                    
                    for i, lead_idx in enumerate(lead_order):
                        row = i // 4 + 1
                        col = i % 4 + 1
                        
                        fig.add_trace(
                            go.Scatter(
                                y=ecg_data[:n_samples, lead_idx],
                                name=f"Lead {lead_idx+1}",
                                line=dict(color='black')
                            ),
                            row=row, col=col
                        )
                    
                    fig.update_layout(
                        title="12-Lead ECG",
                        height=800,
                        width=1000,
                        showlegend=False
                    )
                    
                    # Add grid lines to simulate ECG paper
                    for i in range(1, 4):
                        for j in range(1, 5):
                            fig.update_xaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightpink',
                                row=i, col=j
                            )
                            fig.update_yaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightpink',
                                row=i, col=j
                            )
                    
                    plots['12lead'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
                elif plot_type == 'rhythm':
                    # Rhythm strip
                    lead_idx = int(request.form.get('lead_idx', 0))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            y=ecg_data[:n_samples, lead_idx],
                            name=f"Lead {lead_idx+1}",
                            line=dict(color='black')
                        )
                    )
                    
                    # Add grid lines to simulate ECG paper
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightpink'
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightpink'
                    )
                    
                    fig.update_layout(
                        title=f"Rhythm Strip - Lead {lead_idx+1}",
                        height=300,
                        width=1000,
                        xaxis_title="Sample",
                        yaxis_title="Amplitude"
                    )
                    
                    plots['rhythm'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'visualization/ecg_result.html',
                    plots=plots,
                    filename=filename,
                    plot_type=plot_type,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Error visualizing ECG: {e}")
                flash(f"Error visualizing ECG: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('visualization/ecg.html')

@visualization_bp.route('/explanation', methods=['GET', 'POST'])
@login_required
def explanation():
    """
    Visualize explanation for ECG analysis.
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
                
                # Get explanation parameters
                explanation_method = request.form.get('explanation_method', 'grad_cam')
                
                # Analyze ECG with explanation
                results = system.analyze_ecg(ecg_data, metadata, use_dl=True, explain=True)
                
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
                
                # Explanation plot
                if results.get('explanation') is not None:
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
                        title=f"Explanation Visualization ({explanation_method})",
                        height=400,
                        width=600
                    )
                    
                    plots['explanation'] = json.dumps(explanation_fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'visualization/explanation_result.html',
                    results=results,
                    plots=plots,
                    filename=filename,
                    explanation_method=explanation_method
                )
                
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                flash(f"Error generating explanation: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('visualization/explanation.html')

@visualization_bp.route('/features', methods=['GET', 'POST'])
@login_required
def features():
    """
    Visualize features extracted from ECG data.
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
                
                # Extract features
                features = system.extract_features(ecg_data, metadata)
                
                # Generate plots
                plots = {}
                
                # Feature importance plot
                feature_names = list(features.keys())
                feature_values = list(features.values())
                
                # Sort features by absolute value
                sorted_indices = np.argsort(np.abs(feature_values))[::-1]
                sorted_names = [feature_names[i] for i in sorted_indices[:20]]  # Top 20 features
                sorted_values = [feature_values[i] for i in sorted_indices[:20]]
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_names,
                        y=sorted_values,
                        marker_color='blue'
                    )
                )
                
                fig.update_layout(
                    title="Top 20 Features by Magnitude",
                    height=500,
                    width=1000,
                    xaxis_title="Feature",
                    yaxis_title="Value"
                )
                
                plots['feature_importance'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'visualization/features_result.html',
                    features=features,
                    plots=plots,
                    filename=filename
                )
                
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                flash(f"Error extracting features: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('visualization/features.html')

@visualization_bp.route('/comparison', methods=['GET', 'POST'])
@login_required
def comparison():
    """
    Compare multiple ECG analyses.
    """
    if request.method == 'POST':
        # Get selected result IDs
        result_ids = request.form.getlist('result_ids')
        
        if not result_ids or len(result_ids) < 2:
            flash('Please select at least two results to compare', 'error')
            return redirect(request.url)
        
        try:
            # Get CardioInsight AI system
            from app import get_system
            system = get_system()
            
            # Load results
            results = []
            results_dir = system.config['results_dir']
            
            for result_id in result_ids:
                result_path = os.path.join(results_dir, result_id)
                
                if not os.path.exists(result_path):
                    logger.warning(f"Result {result_id} not found, skipping")
                    continue
                
                with open(result_path, 'r') as f:
                    result = json.load(f)
                    result['id'] = result_id
                    results.append(result)
            
            if len(results) < 2:
                flash('Not enough valid results to compare', 'error')
                return redirect(request.url)
            
            # Generate comparison plots
            plots = {}
            
            # Prediction confidence comparison
            fig = go.Figure()
            
            for result in results:
                fig.add_trace(
                    go.Bar(
                        x=[result.get('id', 'Unknown')],
                        y=[result.get('confidence', 0)],
                        name=result.get('prediction', 'Unknown')
                    )
                )
            
            fig.update_layout(
                title="Prediction Confidence Comparison",
                height=500,
                width=800,
                xaxis_title="Result ID",
                yaxis_title="Confidence"
            )
            
            plots['confidence'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template(
                'visualization/comparison_result.html',
                results=results,
                plots=plots
            )
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            flash(f"Error comparing results: {str(e)}", 'error')
            return redirect(request.url)
    
    # Get list of available results
    from app import get_system
    system = get_system()
    
    results = []
    results_dir = system.config['results_dir']
    
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        for file in result_files[:20]:  # Get 20 most recent
            file_path = os.path.join(results_dir, file)
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    
                results.append({
                    'id': file,
                    'timestamp': result.get('timestamp', ''),
                    'prediction': result.get('prediction', ''),
                    'confidence': result.get('confidence', 0)
                })
            except Exception as e:
                logger.error(f"Error loading result file {file}: {e}")
    
    return render_template('visualization/comparison.html', results=results)

