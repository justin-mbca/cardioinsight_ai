"""
CardioInsight AI - Authentication Module

This module handles user authentication for the CardioInsight AI web application.
"""

import os
import logging
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Mock user database (in a real application, this would be a database)
users = {
    'admin': {
        'password': 'admin123',  # In a real application, this would be hashed
        'role': 'admin'
    },
    'doctor': {
        'password': 'doctor123',
        'role': 'doctor'
    },
    'researcher': {
        'password': 'researcher123',
        'role': 'researcher'
    }
}

def login_required(f):
    """
    Decorator to require login for a route.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    """
    Decorator to require a specific role for a route.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page', 'warning')
                return redirect(url_for('auth.login', next=request.url))
            
            user_id = session['user_id']
            if user_id not in users or users[user_id]['role'] != role:
                flash('You do not have permission to access this page', 'error')
                return redirect(url_for('index'))
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.before_app_request
def load_logged_in_user():
    """
    Load the logged-in user before each request.
    """
    user_id = session.get('user_id')
    
    if user_id is None:
        g.user = None
    else:
        g.user = {
            'id': user_id,
            'role': users.get(user_id, {}).get('role', '')
        }

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login.
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        error = None
        
        if username not in users:
            error = 'Invalid username'
        elif users[username]['password'] != password:
            error = 'Invalid password'
        
        if error is None:
            # Clear the session
            session.clear()
            
            # Store user ID in session
            session['user_id'] = username
            
            # Log login
            logger.info(f"User {username} logged in")
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                return redirect(url_for('dashboard'))
        
        flash(error, 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
def logout():
    """
    Handle user logout.
    """
    # Log logout
    if 'user_id' in session:
        logger.info(f"User {session['user_id']} logged out")
    
    # Clear the session
    session.clear()
    
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/profile')
@login_required
def profile():
    """
    Display user profile.
    """
    user_id = session['user_id']
    user = {
        'id': user_id,
        'role': users.get(user_id, {}).get('role', '')
    }
    
    return render_template('auth/profile.html', user=user)

