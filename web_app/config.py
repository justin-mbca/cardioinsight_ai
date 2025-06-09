"""
CardioInsight AI - Web Application Configuration

This file contains the configuration for the CardioInsight AI web application.
"""

import os
import secrets

class Config:
    """
    Configuration class for the CardioInsight AI web application.
    """
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'mat', 'hea', 'dat', 'npy', 'npz', 'h5', 'hdf5'}
    
    # Database configuration (if needed)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'cardioinsight.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # CardioInsight AI configuration
    CARDIOINSIGHT_CONFIG = {
        'data_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        'models_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
        'results_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'),
        'use_gpu': False,  # Set to True if GPU is available
        'default_model': 'dl_model',
        'logging': {
            'level': 'INFO',
            'file': os.path.join(os.path.dirname(__file__), 'cardioinsight_web.log'),
            'console': True
        }
    }
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

