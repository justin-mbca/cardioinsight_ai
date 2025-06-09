"""
CardioInsight AI - Data Management Module

This module handles data management for the CardioInsight AI web application.
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
data_bp = Blueprint('data', __name__)

def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    
    Parameters:
    -----------
    filename : str
        Name of the file.
        
    Returns:
    --------
    bool
        True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@data_bp.route('/')
@login_required
def index():
    """
    Display the data management page.
    """
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
    
    return render_template('data/index.html', files=files)

@data_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """
    Handle file upload.
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            flash(f'File {filename} uploaded successfully', 'success')
            return redirect(url_for('data.index'))
        else:
            flash('Invalid file type', 'error')
            return redirect(request.url)
    
    return render_template('data/upload.html')

@data_bp.route('/view/<filename>')
@login_required
def view(filename):
    """
    View a file.
    """
    # Secure the filename
    filename = secure_filename(filename)
    
    # Get file path
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads', filename)
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('data.index'))
    
    # Get file extension
    ext = filename.rsplit('.', 1)[1].lower()
    
    # Load file based on extension
    try:
        if ext == 'csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check if it's an ECG file
            if df.shape[1] > 1:
                # Assume first column is time and others are leads
                time_col = df.columns[0]
                lead_cols = df.columns[1:]
                
                # Create plot
                fig = make_subplots(rows=len(lead_cols), cols=1, shared_xaxes=True)
                
                for i, lead in enumerate(lead_cols):
                    fig.add_trace(
                        go.Scatter(
                            x=df[time_col],
                            y=df[lead],
                            name=lead
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    title=f"ECG Data: {filename}",
                    height=100 * len(lead_cols),
                    width=1000
                )
                
                plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'data/view_ecg.html',
                    filename=filename,
                    plot_json=plot_json,
                    df_head=df.head().to_html(classes='table table-striped'),
                    df_shape=df.shape
                )
            else:
                # Regular CSV file
                return render_template(
                    'data/view_csv.html',
                    filename=filename,
                    df_head=df.head().to_html(classes='table table-striped'),
                    df_shape=df.shape
                )
        
        elif ext in ['npy', 'npz']:
            # Load NumPy file
            if ext == 'npy':
                data = np.load(file_path)
            else:
                data = np.load(file_path, allow_pickle=True)
                
                # If it's a dictionary-like object, show keys
                if isinstance(data, np.lib.npyio.NpzFile):
                    return render_template(
                        'data/view_npz.html',
                        filename=filename,
                        keys=list(data.keys()),
                        shapes={k: data[k].shape for k in data.keys()}
                    )
            
            # Check if it's likely an ECG file
            if len(data.shape) == 2:
                # Assume rows are samples and columns are leads
                n_samples, n_leads = data.shape
                
                # Create plot
                fig = make_subplots(rows=min(n_leads, 12), cols=1, shared_xaxes=True)
                
                for i in range(min(n_leads, 12)):
                    fig.add_trace(
                        go.Scatter(
                            y=data[:, i],
                            name=f"Lead {i+1}"
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    title=f"ECG Data: {filename}",
                    height=100 * min(n_leads, 12),
                    width=1000
                )
                
                plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return render_template(
                    'data/view_ecg.html',
                    filename=filename,
                    plot_json=plot_json,
                    data_shape=data.shape
                )
            else:
                # Regular NumPy file
                return render_template(
                    'data/view_numpy.html',
                    filename=filename,
                    data_shape=data.shape,
                    data_type=str(data.dtype)
                )
        
        elif ext in ['mat']:
            # For MAT files, we need scipy.io
            import scipy.io
            
            # Load MAT file
            data = scipy.io.loadmat(file_path)
            
            # Show keys
            return render_template(
                'data/view_mat.html',
                filename=filename,
                keys=list(data.keys()),
                shapes={k: data[k].shape if hasattr(data[k], 'shape') else 'N/A' for k in data.keys() if not k.startswith('__')}
            )
        
        elif ext in ['hea', 'dat']:
            # For WFDB files, we need wfdb
            import wfdb
            
            # Get base name without extension
            base_name = os.path.splitext(file_path)[0]
            
            # Read record
            record = wfdb.rdrecord(base_name)
            
            # Create plot
            fig = make_subplots(rows=record.n_sig, cols=1, shared_xaxes=True)
            
            for i in range(record.n_sig):
                fig.add_trace(
                    go.Scatter(
                        y=record.p_signal[:, i],
                        name=record.sig_name[i]
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                title=f"ECG Data: {filename}",
                height=100 * record.n_sig,
                width=1000
            )
            
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template(
                'data/view_wfdb.html',
                filename=filename,
                plot_json=plot_json,
                record=record.__dict__
            )
        
        else:
            # Unsupported file type for viewing
            flash(f'Cannot preview file of type {ext}', 'warning')
            return redirect(url_for('data.index'))
    
    except Exception as e:
        logger.error(f"Error viewing file {filename}: {e}")
        flash(f"Error viewing file: {str(e)}", 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/download/<filename>')
@login_required
def download(filename):
    """
    Download a file.
    """
    # Secure the filename
    filename = secure_filename(filename)
    
    # Get upload directory
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads')
    
    return send_from_directory(upload_dir, filename, as_attachment=True)

@data_bp.route('/delete/<filename>', methods=['POST'])
@login_required
def delete(filename):
    """
    Delete a file.
    """
    # Secure the filename
    filename = secure_filename(filename)
    
    # Get file path
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads', filename)
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('data.index'))
    
    try:
        # Delete file
        os.remove(file_path)
        
        flash(f'File {filename} deleted successfully', 'success')
        return redirect(url_for('data.index'))
    
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        flash(f"Error deleting file: {str(e)}", 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/preprocess/<filename>', methods=['GET', 'POST'])
@login_required
def preprocess(filename):
    """
    Preprocess a file.
    """
    # Secure the filename
    filename = secure_filename(filename)
    
    # Get file path
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads', filename)
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('data.index'))
    
    if request.method == 'POST':
        try:
            # Get preprocessing parameters
            filter_lowcut = float(request.form.get('filter_lowcut', 0.5))
            filter_highcut = float(request.form.get('filter_highcut', 50.0))
            normalize = request.form.get('normalize', 'false') == 'true'
            remove_baseline = request.form.get('remove_baseline', 'false') == 'true'
            resample_rate = int(request.form.get('resample_rate', 250))
            
            # Get CardioInsight AI system
            from app import get_system
            system = get_system()
            
            # Load ECG data
            ecg_data, metadata = system.load_ecg_data(file_path)
            
            # Add sampling rate if not in metadata
            if 'sampling_rate' not in metadata:
                metadata['sampling_rate'] = float(request.form.get('sampling_rate', 250))
            
            # Preprocess ECG data
            preprocessed_data = system.preprocess_ecg(
                ecg_data,
                metadata,
                filter_lowcut=filter_lowcut,
                filter_highcut=filter_highcut,
                normalize=normalize,
                remove_baseline=remove_baseline,
                resample_rate=resample_rate
            )
            
            # Save preprocessed data
            base_name = os.path.splitext(filename)[0]
            preprocessed_filename = f"{base_name}_preprocessed.npy"
            preprocessed_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'uploads', preprocessed_filename)
            
            np.save(preprocessed_path, preprocessed_data)
            
            flash(f'File {filename} preprocessed successfully. Saved as {preprocessed_filename}', 'success')
            return redirect(url_for('data.view', filename=preprocessed_filename))
        
        except Exception as e:
            logger.error(f"Error preprocessing file {filename}: {e}")
            flash(f"Error preprocessing file: {str(e)}", 'error')
            return redirect(url_for('data.index'))
    
    return render_template('data/preprocess.html', filename=filename)

