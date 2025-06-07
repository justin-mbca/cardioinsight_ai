"""
CardioInsight AI - Dynamic ECG Annotation Module

This module provides tools for automatic annotation of dynamic ECG data (e.g., Holter recordings).
It includes algorithms for QRS detection, rhythm analysis, and anomaly detection in long-term ECG.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import neurokit2 as nk
import warnings
import os
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class QRSDetector:
    """
    Class for QRS complex detection in ECG signals.
    """
    
    def __init__(self, method='neurokit'):
        """
        Initialize the QRS detector.
        
        Parameters:
        -----------
        method : str
            Detection method. Options: 'neurokit', 'pantompkins', 'hamilton', 'engzee'. Default is 'neurokit'.
        """
        self.method = method
        
    def detect(self, ecg_signal, sampling_rate=250):
        """
        Detect QRS complexes in the ECG signal.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
            
        Returns:
        --------
        r_peaks : array-like
            Indices of R-peaks.
        """
        # Check if neurokit2 is installed
        try:
            import neurokit2 as nk
        except ImportError:
            raise ImportError("neurokit2 is required for QRS detection.")
            
        # Detect R-peaks
        if self.method == 'neurokit':
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            r_peaks = r_peaks['ECG_R_Peaks']
        elif self.method == 'pantompkins':
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method='pantompkins')
            r_peaks = r_peaks['ECG_R_Peaks']
        elif self.method == 'hamilton':
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method='hamilton')
            r_peaks = r_peaks['ECG_R_Peaks']
        elif self.method == 'engzee':
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, method='engzee')
            r_peaks = r_peaks['ECG_R_Peaks']
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return r_peaks
    
    def plot_detection(self, ecg_signal, r_peaks, sampling_rate=250, segment=None):
        """
        Plot the ECG signal with detected R-peaks.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        r_peaks : array-like
            Indices of R-peaks.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
        segment : tuple or None
            Time segment to plot in seconds (start, end). If None, plots the entire signal.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create time array
        time = np.arange(len(ecg_signal)) / sampling_rate
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Set segment if provided
        if segment is not None:
            start_idx = int(segment[0] * sampling_rate)
            end_idx = int(segment[1] * sampling_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(ecg_signal), end_idx)
            
            # Plot segment
            ax.plot(time[start_idx:end_idx], ecg_signal[start_idx:end_idx])
            
            # Filter R-peaks within segment
            segment_r_peaks = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
            
            # Plot R-peaks
            ax.scatter(time[segment_r_peaks], ecg_signal[segment_r_peaks], color='red', marker='o')
        else:
            # Plot entire signal
            ax.plot(time, ecg_signal)
            
            # Plot R-peaks
            ax.scatter(time[r_peaks], ecg_signal[r_peaks], color='red', marker='o')
            
        ax.set_title('ECG Signal with Detected R-peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        
        fig.tight_layout()
        return fig


class WaveformSegmenter:
    """
    Class for segmenting ECG waveforms (P, QRS, T).
    """
    
    def __init__(self, method='neurokit'):
        """
        Initialize the waveform segmenter.
        
        Parameters:
        -----------
        method : str
            Segmentation method. Options: 'neurokit', 'custom'. Default is 'neurokit'.
        """
        self.method = method
        
    def segment(self, ecg_signal, r_peaks=None, sampling_rate=250):
        """
        Segment ECG waveforms.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        r_peaks : array-like or None
            Indices of R-peaks. If None, detects R-peaks.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
            
        Returns:
        --------
        waves : dict
            Dictionary containing the delineated waveforms.
        """
        # Check if neurokit2 is installed
        try:
            import neurokit2 as nk
        except ImportError:
            raise ImportError("neurokit2 is required for waveform segmentation.")
            
        # Detect R-peaks if not provided
        if r_peaks is None:
            qrs_detector = QRSDetector(method='neurokit')
            r_peaks = qrs_detector.detect(ecg_signal, sampling_rate)
            
        # Segment waveforms
        if self.method == 'neurokit':
            # Delineate
            _, waves = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=sampling_rate)
            
            # Clean up waves dictionary
            waves = {k: v for k, v in waves.items() if v is not None}
        elif self.method == 'custom':
            # Custom segmentation logic
            waves = self._custom_segment(ecg_signal, r_peaks, sampling_rate)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return waves
    
    def _custom_segment(self, ecg_signal, r_peaks, sampling_rate):
        """
        Custom segmentation logic.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        r_peaks : array-like
            Indices of R-peaks.
        sampling_rate : int
            Sampling rate in Hz.
            
        Returns:
        --------
        waves : dict
            Dictionary containing the delineated waveforms.
        """
        # Initialize waves dictionary
        waves = {
            'ECG_P_Peaks': [],
            'ECG_Q_Peaks': [],
            'ECG_S_Peaks': [],
            'ECG_T_Peaks': [],
            'ECG_P_Onsets': [],
            'ECG_P_Offsets': [],
            'ECG_QRS_Onsets': [],
            'ECG_QRS_Offsets': [],
            'ECG_T_Onsets': [],
            'ECG_T_Offsets': []
        }
        
        # Process each R-peak
        for r_peak in r_peaks:
            # Define search windows
            # Q wave: 50-10 ms before R-peak
            q_window_start = max(0, r_peak - int(0.05 * sampling_rate))
            q_window_end = max(0, r_peak - int(0.01 * sampling_rate))
            
            # S wave: 10-50 ms after R-peak
            s_window_start = min(len(ecg_signal) - 1, r_peak + int(0.01 * sampling_rate))
            s_window_end = min(len(ecg_signal) - 1, r_peak + int(0.05 * sampling_rate))
            
            # P wave: 200-50 ms before R-peak
            p_window_start = max(0, r_peak - int(0.2 * sampling_rate))
            p_window_end = max(0, r_peak - int(0.05 * sampling_rate))
            
            # T wave: 50-350 ms after R-peak
            t_window_start = min(len(ecg_signal) - 1, r_peak + int(0.05 * sampling_rate))
            t_window_end = min(len(ecg_signal) - 1, r_peak + int(0.35 * sampling_rate))
            
            # Find Q wave (minimum in Q window)
            if q_window_end > q_window_start:
                q_segment = ecg_signal[q_window_start:q_window_end]
                q_peak = q_window_start + np.argmin(q_segment)
                waves['ECG_Q_Peaks'].append(q_peak)
            
            # Find S wave (minimum in S window)
            if s_window_end > s_window_start:
                s_segment = ecg_signal[s_window_start:s_window_end]
                s_peak = s_window_start + np.argmin(s_segment)
                waves['ECG_S_Peaks'].append(s_peak)
            
            # Find P wave (maximum in P window)
            if p_window_end > p_window_start:
                p_segment = ecg_signal[p_window_start:p_window_end]
                p_peak = p_window_start + np.argmax(p_segment)
                waves['ECG_P_Peaks'].append(p_peak)
                
                # P wave onset/offset
                # Simple approach: fixed window around P peak
                waves['ECG_P_Onsets'].append(max(0, p_peak - int(0.04 * sampling_rate)))
                waves['ECG_P_Offsets'].append(min(len(ecg_signal) - 1, p_peak + int(0.04 * sampling_rate)))
            
            # Find T wave (maximum in T window)
            if t_window_end > t_window_start:
                t_segment = ecg_signal[t_window_start:t_window_end]
                t_peak = t_window_start + np.argmax(t_segment)
                waves['ECG_T_Peaks'].append(t_peak)
                
                # T wave onset/offset
                # Simple approach: fixed window around T peak
                waves['ECG_T_Onsets'].append(max(0, t_peak - int(0.05 * sampling_rate)))
                waves['ECG_T_Offsets'].append(min(len(ecg_signal) - 1, t_peak + int(0.05 * sampling_rate)))
            
            # QRS onset/offset
            if 'ECG_Q_Peaks' in waves and len(waves['ECG_Q_Peaks']) > 0 and 'ECG_S_Peaks' in waves and len(waves['ECG_S_Peaks']) > 0:
                q_peak = waves['ECG_Q_Peaks'][-1]
                s_peak = waves['ECG_S_Peaks'][-1]
                waves['ECG_QRS_Onsets'].append(max(0, q_peak - int(0.02 * sampling_rate)))
                waves['ECG_QRS_Offsets'].append(min(len(ecg_signal) - 1, s_peak + int(0.02 * sampling_rate)))
            else:
                # Fallback if Q or S not found
                waves['ECG_QRS_Onsets'].append(max(0, r_peak - int(0.05 * sampling_rate)))
                waves['ECG_QRS_Offsets'].append(min(len(ecg_signal) - 1, r_peak + int(0.05 * sampling_rate)))
                
        # Convert lists to numpy arrays
        for key in waves:
            waves[key] = np.array(waves[key])
            
        return waves
    
    def plot_segmentation(self, ecg_signal, waves, sampling_rate=250, segment=None):
        """
        Plot the ECG signal with segmented waveforms.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        waves : dict
            Dictionary containing the delineated waveforms.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
        segment : tuple or None
            Time segment to plot in seconds (start, end). If None, plots the entire signal.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create time array
        time = np.arange(len(ecg_signal)) / sampling_rate
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Set segment if provided
        if segment is not None:
            start_idx = int(segment[0] * sampling_rate)
            end_idx = int(segment[1] * sampling_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(ecg_signal), end_idx)
            
            # Plot segment
            ax.plot(time[start_idx:end_idx], ecg_signal[start_idx:end_idx])
            
            # Filter waves within segment
            segment_waves = {}
            for wave_type, indices in waves.items():
                if indices is not None:
                    segment_waves[wave_type] = indices[(indices >= start_idx) & (indices < end_idx)]
                else:
                    segment_waves[wave_type] = None
        else:
            # Plot entire signal
            ax.plot(time, ecg_signal)
            segment_waves = waves
            
        # Plot waves
        wave_colors = {
            'ECG_P_Peaks': 'green',
            'ECG_Q_Peaks': 'orange',
            'ECG_R_Peaks': 'red',
            'ECG_S_Peaks': 'purple',
            'ECG_T_Peaks': 'blue',
            'ECG_P_Onsets': 'lightgreen',
            'ECG_P_Offsets': 'darkgreen',
            'ECG_QRS_Onsets': 'salmon',
            'ECG_QRS_Offsets': 'darkred',
            'ECG_T_Onsets': 'lightblue',
            'ECG_T_Offsets': 'darkblue'
        }
        
        for wave_type, indices in segment_waves.items():
            if indices is not None and len(indices) > 0 and wave_type in wave_colors:
                if 'Peaks' in wave_type:
                    ax.scatter(time[indices], ecg_signal[indices], color=wave_colors[wave_type], marker='o', label=wave_type)
                else:
                    # For onsets and offsets, plot vertical lines
                    for idx in indices:
                        if idx < len(time):
                            ax.axvline(x=time[idx], color=wave_colors[wave_type], linestyle='--', alpha=0.5)
            
        ax.set_title('ECG Signal with Segmented Waveforms')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        
        fig.tight_layout()
        return fig


class RhythmAnalyzer:
    """
    Class for analyzing ECG rhythms.
    """
    
    def __init__(self):
        """
        Initialize the rhythm analyzer.
        """
        pass
    
    def analyze_rr_intervals(self, r_peaks, sampling_rate=250):
        """
        Analyze RR intervals.
        
        Parameters:
        -----------
        r_peaks : array-like
            Indices of R-peaks.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
            
        Returns:
        --------
        results : dict
            Dictionary containing rhythm analysis results.
        """
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # in ms
        
        # Calculate heart rate
        heart_rate = 60000 / rr_intervals  # in bpm
        
        # Calculate heart rate variability metrics
        hrv_metrics = self._calculate_hrv_metrics(rr_intervals)
        
        # Detect irregular rhythms
        irregular_indices = self._detect_irregular_rhythms(rr_intervals)
        
        # Compile results
        results = {
            'rr_intervals': rr_intervals,
            'heart_rate': heart_rate,
            'mean_hr': np.mean(heart_rate),
            'std_hr': np.std(heart_rate),
            'min_hr': np.min(heart_rate),
            'max_hr': np.max(heart_rate),
            'hrv_metrics': hrv_metrics,
            'irregular_indices': irregular_indices
        }
        
        return results
    
    def _calculate_hrv_metrics(self, rr_intervals):
        """
        Calculate heart rate variability metrics.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in ms.
            
        Returns:
        --------
        hrv_metrics : dict
            Dictionary containing HRV metrics.
        """
        # Time-domain metrics
        sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
        
        # Calculate pNN50
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
        
        # Compile metrics
        hrv_metrics = {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': nn50,
            'pnn50': pnn50
        }
        
        return hrv_metrics
    
    def _detect_irregular_rhythms(self, rr_intervals):
        """
        Detect irregular rhythms.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in ms.
            
        Returns:
        --------
        irregular_indices : array-like
            Indices of irregular RR intervals.
        """
        # Calculate median RR interval
        median_rr = np.median(rr_intervals)
        
        # Define irregularity threshold (e.g., 20% deviation from median)
        threshold = 0.2 * median_rr
        
        # Find irregular intervals
        irregular_indices = np.where(np.abs(rr_intervals - median_rr) > threshold)[0]
        
        return irregular_indices
    
    def classify_rhythm(self, rr_intervals, hrv_metrics):
        """
        Classify heart rhythm.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in ms.
        hrv_metrics : dict
            Dictionary containing HRV metrics.
            
        Returns:
        --------
        rhythm_type : str
            Classified rhythm type.
        confidence : float
            Confidence score.
        """
        # Calculate mean heart rate
        mean_hr = 60000 / np.mean(rr_intervals)
        
        # Calculate irregularity
        irregularity = hrv_metrics['sdnn'] / np.mean(rr_intervals)
        
        # Simple rule-based classification
        if mean_hr < 60:
            rhythm_type = 'Bradycardia'
            confidence = min(1.0, (60 - mean_hr) / 15)
        elif mean_hr > 100:
            rhythm_type = 'Tachycardia'
            confidence = min(1.0, (mean_hr - 100) / 40)
        elif irregularity > 0.2:
            rhythm_type = 'Irregular Rhythm'
            confidence = min(1.0, irregularity / 0.4)
        else:
            rhythm_type = 'Normal Sinus Rhythm'
            confidence = 1.0 - irregularity
            
        return rhythm_type, confidence
    
    def plot_rr_intervals(self, rr_intervals, irregular_indices=None):
        """
        Plot RR intervals.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in ms.
        irregular_indices : array-like or None
            Indices of irregular RR intervals. If None, doesn't highlight irregularities.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot RR intervals
        ax1.plot(rr_intervals)
        ax1.set_title('RR Intervals')
        ax1.set_xlabel('Beat Number')
        ax1.set_ylabel('RR Interval (ms)')
        ax1.grid(True)
        
        # Highlight irregular intervals
        if irregular_indices is not None and len(irregular_indices) > 0:
            ax1.scatter(irregular_indices, rr_intervals[irregular_indices], color='red', marker='o')
            
        # Plot heart rate
        heart_rate = 60000 / rr_intervals
        ax2.plot(heart_rate)
        ax2.set_title('Heart Rate')
        ax2.set_xlabel('Beat Number')
        ax2.set_ylabel('Heart Rate (bpm)')
        ax2.grid(True)
        
        # Highlight irregular beats in heart rate
        if irregular_indices is not None and len(irregular_indices) > 0:
            ax2.scatter(irregular_indices, heart_rate[irregular_indices], color='red', marker='o')
            
        fig.tight_layout()
        return fig
    
    def plot_poincare(self, rr_intervals):
        """
        Plot Poincaré plot of RR intervals.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in ms.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create Poincaré plot
        ax.scatter(rr_intervals[:-1], rr_intervals[1:], alpha=0.5)
        ax.set_title('Poincaré Plot')
        ax.set_xlabel('RR(n) (ms)')
        ax.set_ylabel('RR(n+1) (ms)')
        ax.grid(True)
        
        # Add identity line
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        ax.plot([min_rr, max_rr], [min_rr, max_rr], 'k--')
        
        # Make plot square
        ax.set_aspect('equal')
        
        fig.tight_layout()
        return fig


class AnomalyDetector:
    """
    Class for detecting anomalies in ECG signals.
    """
    
    def __init__(self, method='isolation_forest'):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        method : str
            Detection method. Options: 'isolation_forest', 'dbscan', 'autoencoder'. Default is 'isolation_forest'.
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, ecg_signal, r_peaks, sampling_rate=250):
        """
        Extract features from ECG signal.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        r_peaks : array-like
            Indices of R-peaks.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # in ms
        
        # Calculate heart rate
        heart_rate = 60000 / rr_intervals  # in bpm
        
        # Extract beat-to-beat features
        features = []
        
        for i in range(1, len(r_peaks) - 1):
            # Current RR interval
            curr_rr = rr_intervals[i-1]
            
            # Previous and next RR intervals
            prev_rr = rr_intervals[i-2] if i > 1 else curr_rr
            next_rr = rr_intervals[i] if i < len(rr_intervals) else curr_rr
            
            # RR interval ratios
            rr_ratio_prev = curr_rr / prev_rr if prev_rr > 0 else 1
            rr_ratio_next = curr_rr / next_rr if next_rr > 0 else 1
            
            # Extract beat waveform
            beat_start = r_peaks[i-1]
            beat_end = r_peaks[i]
            beat_length = beat_end - beat_start
            
            # Resample beat to fixed length (e.g., 100 samples)
            if beat_length > 1:
                beat = ecg_signal[beat_start:beat_end]
                x_old = np.linspace(0, 1, len(beat))
                x_new = np.linspace(0, 1, 20)  # Resample to 20 points
                
                try:
                    interp_func = interp1d(x_old, beat)
                    resampled_beat = interp_func(x_new)
                    
                    # Normalize beat
                    resampled_beat = (resampled_beat - np.mean(resampled_beat)) / (np.std(resampled_beat) + 1e-6)
                    
                    # Combine features
                    beat_features = [
                        curr_rr,
                        heart_rate[i-1],
                        rr_ratio_prev,
                        rr_ratio_next
                    ]
                    
                    # Add resampled beat
                    beat_features.extend(resampled_beat)
                    
                    features.append(beat_features)
                except:
                    # Skip problematic beats
                    continue
                    
        return np.array(features)
    
    def fit(self, features):
        """
        Fit the anomaly detection model.
        
        Parameters:
        -----------
        features : array-like
            Features extracted from ECG signal.
            
        Returns:
        --------
        self : object
            Fitted model.
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        if self.method == 'isolation_forest':
            # Isolation Forest
            self.model = IsolationForest(contamination=0.05, random_state=42)
            self.model.fit(scaled_features)
        elif self.method == 'dbscan':
            # DBSCAN
            self.model = DBSCAN(eps=0.5, min_samples=5)
            self.model.fit(scaled_features)
        elif self.method == 'autoencoder':
            # Autoencoder
            self.model = self._build_autoencoder(scaled_features.shape[1])
            
            # Split data for training
            train_features, val_features = train_test_split(scaled_features, test_size=0.2, random_state=42)
            
            # Train autoencoder
            self.model.fit(
                train_features, train_features,
                epochs=50,
                batch_size=32,
                validation_data=(val_features, val_features),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ],
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return self
    
    def _build_autoencoder(self, input_dim):
        """
        Build autoencoder model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension.
            
        Returns:
        --------
        model : Model
            Autoencoder model.
        """
        # Define encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Define decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Create autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def detect(self, features):
        """
        Detect anomalies.
        
        Parameters:
        -----------
        features : array-like
            Features extracted from ECG signal.
            
        Returns:
        --------
        anomaly_scores : array-like
            Anomaly scores.
        anomaly_indices : array-like
            Indices of detected anomalies.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        if self.method == 'isolation_forest':
            # Isolation Forest
            # Decision function: negative score = anomaly
            anomaly_scores = -self.model.decision_function(scaled_features)
            anomaly_indices = np.where(self.model.predict(scaled_features) == -1)[0]
        elif self.method == 'dbscan':
            # DBSCAN
            # Predict on new data
            labels = self.model.fit_predict(scaled_features)
            
            # Anomalies are labeled as -1
            anomaly_scores = np.zeros_like(labels, dtype=float)
            anomaly_scores[labels == -1] = 1.0
            anomaly_indices = np.where(labels == -1)[0]
        elif self.method == 'autoencoder':
            # Autoencoder
            # Reconstruction error as anomaly score
            reconstructions = self.model.predict(scaled_features)
            mse = np.mean(np.square(scaled_features - reconstructions), axis=1)
            
            # Normalize scores
            anomaly_scores = (mse - np.min(mse)) / (np.max(mse) - np.min(mse) + 1e-10)
            
            # Threshold for anomaly detection (e.g., top 5%)
            threshold = np.percentile(anomaly_scores, 95)
            anomaly_indices = np.where(anomaly_scores > threshold)[0]
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return anomaly_scores, anomaly_indices
    
    def plot_anomalies(self, ecg_signal, r_peaks, anomaly_indices, sampling_rate=250):
        """
        Plot ECG signal with detected anomalies.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        r_peaks : array-like
            Indices of R-peaks.
        anomaly_indices : array-like
            Indices of detected anomalies.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create time array
        time = np.arange(len(ecg_signal)) / sampling_rate
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot ECG signal
        ax.plot(time, ecg_signal)
        
        # Plot R-peaks
        ax.scatter(time[r_peaks], ecg_signal[r_peaks], color='blue', marker='o', label='R-peaks')
        
        # Plot anomalies
        anomaly_r_peaks = r_peaks[anomaly_indices]
        ax.scatter(time[anomaly_r_peaks], ecg_signal[anomaly_r_peaks], color='red', marker='x', s=100, label='Anomalies')
        
        ax.set_title('ECG Signal with Detected Anomalies')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        
        fig.tight_layout()
        return fig


class HolterAnalyzer:
    """
    Class for analyzing Holter ECG recordings.
    """
    
    def __init__(self):
        """
        Initialize the Holter analyzer.
        """
        self.qrs_detector = QRSDetector()
        self.waveform_segmenter = WaveformSegmenter()
        self.rhythm_analyzer = RhythmAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
    def analyze(self, ecg_signal, sampling_rate=250, window_size=3600):
        """
        Analyze Holter ECG recording.
        
        Parameters:
        -----------
        ecg_signal : array-like
            ECG signal.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
        window_size : int
            Analysis window size in seconds. Default is 3600 (1 hour).
            
        Returns:
        --------
        results : dict
            Dictionary containing analysis results.
        """
        # Initialize results
        results = {
            'r_peaks': [],
            'waves': {},
            'rhythm_analysis': {},
            'anomalies': [],
            'hourly_stats': []
        }
        
        # Detect R-peaks
        r_peaks = self.qrs_detector.detect(ecg_signal, sampling_rate)
        results['r_peaks'] = r_peaks
        
        # Segment waveforms
        waves = self.waveform_segmenter.segment(ecg_signal, r_peaks, sampling_rate)
        results['waves'] = waves
        
        # Analyze rhythm
        rhythm_results = self.rhythm_analyzer.analyze_rr_intervals(r_peaks, sampling_rate)
        results['rhythm_analysis'] = rhythm_results
        
        # Extract features for anomaly detection
        features = self.anomaly_detector.extract_features(ecg_signal, r_peaks, sampling_rate)
        
        # Fit anomaly detector and detect anomalies
        if len(features) > 0:
            self.anomaly_detector.fit(features)
            anomaly_scores, anomaly_indices = self.anomaly_detector.detect(features)
            results['anomalies'] = {
                'scores': anomaly_scores,
                'indices': anomaly_indices
            }
        
        # Analyze by hour
        total_duration = len(ecg_signal) / sampling_rate
        n_windows = int(np.ceil(total_duration / window_size))
        
        for i in range(n_windows):
            # Define window
            start_time = i * window_size
            end_time = min((i + 1) * window_size, total_duration)
            
            # Convert to samples
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            
            # Get window data
            window_signal = ecg_signal[start_sample:end_sample]
            
            # Find R-peaks in window
            window_r_peaks_indices = np.where((r_peaks >= start_sample) & (r_peaks < end_sample))[0]
            window_r_peaks = r_peaks[window_r_peaks_indices] - start_sample
            
            # Skip if no R-peaks in window
            if len(window_r_peaks) < 2:
                continue
                
            # Analyze rhythm in window
            window_rhythm_results = self.rhythm_analyzer.analyze_rr_intervals(window_r_peaks, sampling_rate)
            
            # Classify rhythm
            rhythm_type, confidence = self.rhythm_analyzer.classify_rhythm(
                window_rhythm_results['rr_intervals'],
                window_rhythm_results['hrv_metrics']
            )
            
            # Store hourly stats
            hourly_stat = {
                'start_time': start_time,
                'end_time': end_time,
                'mean_hr': window_rhythm_results['mean_hr'],
                'min_hr': window_rhythm_results['min_hr'],
                'max_hr': window_rhythm_results['max_hr'],
                'sdnn': window_rhythm_results['hrv_metrics']['sdnn'],
                'rmssd': window_rhythm_results['hrv_metrics']['rmssd'],
                'rhythm_type': rhythm_type,
                'rhythm_confidence': confidence
            }
            
            results['hourly_stats'].append(hourly_stat)
            
        return results
    
    def generate_report(self, results):
        """
        Generate a summary report from analysis results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing analysis results.
            
        Returns:
        --------
        report : dict
            Dictionary containing report summary.
        """
        # Extract key metrics
        r_peaks = results['r_peaks']
        rhythm_analysis = results['rhythm_analysis']
        hourly_stats = results['hourly_stats']
        
        # Calculate overall statistics
        total_beats = len(r_peaks)
        mean_hr = rhythm_analysis['mean_hr']
        min_hr = rhythm_analysis['min_hr']
        max_hr = rhythm_analysis['max_hr']
        sdnn = rhythm_analysis['hrv_metrics']['sdnn']
        rmssd = rhythm_analysis['hrv_metrics']['rmssd']
        
        # Count rhythm types
        rhythm_counts = {}
        for stat in hourly_stats:
            rhythm_type = stat['rhythm_type']
            if rhythm_type in rhythm_counts:
                rhythm_counts[rhythm_type] += 1
            else:
                rhythm_counts[rhythm_type] = 1
                
        # Find predominant rhythm
        predominant_rhythm = max(rhythm_counts.items(), key=lambda x: x[1])[0] if rhythm_counts else 'Unknown'
        
        # Count anomalies
        n_anomalies = len(results['anomalies']['indices']) if 'indices' in results['anomalies'] else 0
        
        # Compile report
        report = {
            'total_duration_hours': len(hourly_stats),
            'total_beats': total_beats,
            'mean_heart_rate': mean_hr,
            'min_heart_rate': min_hr,
            'max_heart_rate': max_hr,
            'heart_rate_variability': {
                'sdnn': sdnn,
                'rmssd': rmssd,
                'pnn50': rhythm_analysis['hrv_metrics']['pnn50']
            },
            'predominant_rhythm': predominant_rhythm,
            'rhythm_distribution': rhythm_counts,
            'anomalies_detected': n_anomalies,
            'hourly_summary': hourly_stats
        }
        
        return report
    
    def plot_hourly_trends(self, results):
        """
        Plot hourly trends from analysis results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing analysis results.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Extract hourly stats
        hourly_stats = results['hourly_stats']
        
        if not hourly_stats:
            raise ValueError("No hourly statistics available.")
            
        # Extract data
        hours = [stat['start_time'] / 3600 for stat in hourly_stats]
        mean_hr = [stat['mean_hr'] for stat in hourly_stats]
        min_hr = [stat['min_hr'] for stat in hourly_stats]
        max_hr = [stat['max_hr'] for stat in hourly_stats]
        sdnn = [stat['sdnn'] for stat in hourly_stats]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot heart rate trends
        ax1.plot(hours, mean_hr, 'b-', label='Mean HR')
        ax1.fill_between(hours, min_hr, max_hr, color='blue', alpha=0.2, label='HR Range')
        ax1.set_title('Hourly Heart Rate Trends')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Heart Rate (bpm)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot HRV trends
        ax2.plot(hours, sdnn, 'g-', label='SDNN')
        ax2.set_title('Hourly Heart Rate Variability Trends')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('SDNN (ms)')
        ax2.grid(True)
        ax2.legend()
        
        fig.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Check if neurokit2 is installed
    try:
        import neurokit2 as nk
    except ImportError:
        print("neurokit2 is required for this module. Install it with: pip install neurokit2")
        exit()
        
    # Generate synthetic ECG data
    sampling_rate = 250
    duration = 10  # seconds
    
    # Generate ECG signal using neurokit2
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)
    
    # QRS detection
    qrs_detector = QRSDetector()
    r_peaks = qrs_detector.detect(ecg, sampling_rate)
    
    print(f"Detected {len(r_peaks)} R-peaks")
    
    # Waveform segmentation
    segmenter = WaveformSegmenter()
    waves = segmenter.segment(ecg, r_peaks, sampling_rate)
    
    print("Segmented waveforms:")
    for wave_type, indices in waves.items():
        if indices is not None:
            print(f"  {wave_type}: {len(indices)} points")
            
    # Rhythm analysis
    analyzer = RhythmAnalyzer()
    rhythm_results = analyzer.analyze_rr_intervals(r_peaks, sampling_rate)
    
    print("\nRhythm analysis:")
    print(f"  Mean heart rate: {rhythm_results['mean_hr']:.1f} bpm")
    print(f"  SDNN: {rhythm_results['hrv_metrics']['sdnn']:.1f} ms")
    print(f"  RMSSD: {rhythm_results['hrv_metrics']['rmssd']:.1f} ms")
    
    # Anomaly detection
    detector = AnomalyDetector()
    features = detector.extract_features(ecg, r_peaks, sampling_rate)
    
    if len(features) > 0:
        print(f"\nExtracted {features.shape[1]} features from {features.shape[0]} beats")
        
        detector.fit(features)
        anomaly_scores, anomaly_indices = detector.detect(features)
        
        print(f"Detected {len(anomaly_indices)} anomalies")
    else:
        print("\nNot enough beats for feature extraction")

