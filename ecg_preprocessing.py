"""
CardioInsight AI - ECG Preprocessing Module

This module provides functions for loading and preprocessing ECG data.
It supports various ECG data formats and implements common preprocessing techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import wfdb  # For reading PhysioNet data formats

class ECGPreprocessor:
    """
    Class for loading and preprocessing ECG data.
    """
    
    def __init__(self, sampling_rate=500):
        """
        Initialize the ECG preprocessor.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the ECG data in Hz. Default is 500 Hz.
        """
        self.sampling_rate = sampling_rate
        
    def load_data(self, file_path, format_type='wfdb'):
        """
        Load ECG data from file.
        
        Parameters:
        -----------
        file_path : str
            Path to the ECG data file.
        format_type : str
            Format of the ECG data. Options: 'wfdb', 'csv', 'mat'.
            Default is 'wfdb' (PhysioNet format).
            
        Returns:
        --------
        signals : ndarray
            ECG signals with shape (n_samples, n_leads).
        header : dict
            Header information including lead names, patient info, etc.
        """
        if format_type == 'wfdb':
            # Load PhysioNet format data
            record = wfdb.rdrecord(file_path)
            signals = record.p_signal
            header = {
                'lead_names': record.sig_name,
                'fs': record.fs,
                'n_leads': record.n_sig,
                'n_samples': record.sig_len,
                'patient_info': record.comments
            }
            return signals, header
        
        elif format_type == 'csv':
            # Load CSV format data
            df = pd.read_csv(file_path)
            # Assuming first column is time and others are ECG leads
            signals = df.iloc[:, 1:].values
            lead_names = df.columns[1:]
            header = {
                'lead_names': lead_names,
                'fs': self.sampling_rate,
                'n_leads': len(lead_names),
                'n_samples': len(signals)
            }
            return signals, header
        
        elif format_type == 'mat':
            # Load MATLAB format data
            from scipy.io import loadmat
            mat = loadmat(file_path)
            # Structure depends on specific .mat file format
            # This is a generic approach, might need adjustment
            if 'ECG' in mat:
                signals = mat['ECG']
                header = {
                    'fs': self.sampling_rate,
                    'n_leads': signals.shape[1],
                    'n_samples': signals.shape[0]
                }
                return signals, header
            else:
                raise ValueError("Could not find ECG data in .mat file")
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def filter_signal(self, signal_data, lowcut=0.5, highcut=50.0, order=5):
        """
        Apply bandpass filter to remove noise.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        lowcut : float
            Low cutoff frequency in Hz. Default is 0.5 Hz.
        highcut : float
            High cutoff frequency in Hz. Default is 50.0 Hz.
        order : int
            Filter order. Default is 5.
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered ECG signal.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter
        if signal_data.ndim == 1:
            # Single lead
            filtered_signal = signal.filtfilt(b, a, signal_data)
        else:
            # Multiple leads
            filtered_signal = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered_signal[:, i] = signal.filtfilt(b, a, signal_data[:, i])
                
        return filtered_signal
    
    def remove_baseline_wander(self, signal_data, window_size=500):
        """
        Remove baseline wander using median filter.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        window_size : int
            Window size for median filter. Default is 500 samples.
            
        Returns:
        --------
        corrected_signal : ndarray
            ECG signal with baseline wander removed.
        """
        if signal_data.ndim == 1:
            # Single lead
            baseline = signal.medfilt(signal_data, window_size)
            corrected_signal = signal_data - baseline
        else:
            # Multiple leads
            corrected_signal = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                baseline = signal.medfilt(signal_data[:, i], window_size)
                corrected_signal[:, i] = signal_data[:, i] - baseline
                
        return corrected_signal
    
    def normalize_signal(self, signal_data, method='minmax'):
        """
        Normalize ECG signal.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        method : str
            Normalization method. Options: 'minmax', 'zscore'. Default is 'minmax'.
            
        Returns:
        --------
        normalized_signal : ndarray
            Normalized ECG signal.
        """
        if method == 'minmax':
            if signal_data.ndim == 1:
                # Single lead
                min_val = np.min(signal_data)
                max_val = np.max(signal_data)
                normalized_signal = (signal_data - min_val) / (max_val - min_val)
            else:
                # Multiple leads
                normalized_signal = np.zeros_like(signal_data)
                for i in range(signal_data.shape[1]):
                    min_val = np.min(signal_data[:, i])
                    max_val = np.max(signal_data[:, i])
                    normalized_signal[:, i] = (signal_data[:, i] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            if signal_data.ndim == 1:
                # Single lead
                mean_val = np.mean(signal_data)
                std_val = np.std(signal_data)
                normalized_signal = (signal_data - mean_val) / std_val
            else:
                # Multiple leads
                normalized_signal = np.zeros_like(signal_data)
                for i in range(signal_data.shape[1]):
                    mean_val = np.mean(signal_data[:, i])
                    std_val = np.std(signal_data[:, i])
                    normalized_signal[:, i] = (signal_data[:, i] - mean_val) / std_val
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return normalized_signal
    
    def detect_r_peaks(self, signal_data, lead_idx=0):
        """
        Detect R peaks in ECG signal using Pan-Tompkins algorithm.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        lead_idx : int
            Index of the lead to use for R peak detection. Default is 0.
            
        Returns:
        --------
        r_peaks : ndarray
            Indices of R peaks.
        """
        # Extract the specified lead
        if signal_data.ndim > 1:
            lead = signal_data[:, lead_idx]
        else:
            lead = signal_data
            
        # Apply bandpass filter to enhance QRS complex
        filtered = self.filter_signal(lead, lowcut=5.0, highcut=15.0)
        
        # Differentiate the signal
        diff = np.diff(filtered)
        
        # Square the signal
        squared = diff ** 2
        
        # Apply moving average filter
        window_size = int(0.15 * self.sampling_rate)  # 150 ms window
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        integrated = signal.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks
        r_peaks, _ = signal.find_peaks(integrated, height=0.5*np.max(integrated), distance=0.2*self.sampling_rate)
        
        # Adjust peak positions to original signal
        for i in range(len(r_peaks)):
            # Find the maximum in a small window around the detected peak
            window_start = max(0, r_peaks[i] - 10)
            window_end = min(len(lead), r_peaks[i] + 10)
            r_peaks[i] = window_start + np.argmax(lead[window_start:window_end])
            
        return r_peaks
    
    def segment_heartbeats(self, signal_data, r_peaks, window_size=250):
        """
        Segment ECG signal into individual heartbeats.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        r_peaks : ndarray
            Indices of R peaks.
        window_size : int
            Half-window size around R peak in samples. Default is 250 samples.
            
        Returns:
        --------
        heartbeats : ndarray
            Segmented heartbeats with shape (n_beats, window_size*2, n_leads).
        """
        n_samples = signal_data.shape[0]
        
        if signal_data.ndim == 1:
            # Single lead
            n_leads = 1
            signal_data = signal_data.reshape(-1, 1)
        else:
            # Multiple leads
            n_leads = signal_data.shape[1]
            
        # Initialize array for heartbeats
        heartbeats = []
        
        # Extract heartbeats
        for peak in r_peaks:
            # Define window boundaries
            start = peak - window_size
            end = peak + window_size
            
            # Skip if window extends beyond signal boundaries
            if start < 0 or end >= n_samples:
                continue
                
            # Extract heartbeat
            heartbeat = signal_data[start:end, :]
            heartbeats.append(heartbeat)
            
        return np.array(heartbeats)
    
    def preprocess_ecg(self, signal_data, filter_ecg=True, remove_baseline=True, normalize=True):
        """
        Apply full preprocessing pipeline to ECG signal.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        filter_ecg : bool
            Whether to apply bandpass filter. Default is True.
        remove_baseline : bool
            Whether to remove baseline wander. Default is True.
        normalize : bool
            Whether to normalize the signal. Default is True.
            
        Returns:
        --------
        processed_signal : ndarray
            Preprocessed ECG signal.
        """
        processed_signal = signal_data.copy()
        
        if filter_ecg:
            processed_signal = self.filter_signal(processed_signal)
            
        if remove_baseline:
            processed_signal = self.remove_baseline_wander(processed_signal)
            
        if normalize:
            processed_signal = self.normalize_signal(processed_signal)
            
        return processed_signal
    
    def plot_ecg(self, signal_data, header=None, lead_idx=0, r_peaks=None, title="ECG Signal"):
        """
        Plot ECG signal.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        header : dict
            Header information including lead names.
        lead_idx : int
            Index of the lead to plot. Default is 0.
        r_peaks : ndarray
            Indices of R peaks to mark on the plot.
        title : str
            Plot title. Default is "ECG Signal".
        """
        plt.figure(figsize=(12, 4))
        
        # Extract the specified lead
        if signal_data.ndim > 1:
            lead = signal_data[:, lead_idx]
            lead_name = header['lead_names'][lead_idx] if header and 'lead_names' in header else f"Lead {lead_idx}"
        else:
            lead = signal_data
            lead_name = "ECG"
            
        # Create time axis
        fs = header['fs'] if header and 'fs' in header else self.sampling_rate
        time = np.arange(len(lead)) / fs
        
        # Plot signal
        plt.plot(time, lead)
        
        # Mark R peaks if provided
        if r_peaks is not None:
            plt.plot(r_peaks/fs, lead[r_peaks], 'ro')
            
        plt.title(f"{title} - {lead_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_all_leads(self, signal_data, header=None, title="12-Lead ECG"):
        """
        Plot all ECG leads.
        
        Parameters:
        -----------
        signal_data : ndarray
            ECG signal data.
        header : dict
            Header information including lead names.
        title : str
            Plot title. Default is "12-Lead ECG".
        """
        if signal_data.ndim == 1:
            # Single lead
            return self.plot_ecg(signal_data, header, title=title)
            
        n_leads = signal_data.shape[1]
        
        # Determine grid layout
        n_rows = int(np.ceil(n_leads / 3))
        n_cols = min(n_leads, 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
        axes = axes.flatten()
        
        # Create time axis
        fs = header['fs'] if header and 'fs' in header else self.sampling_rate
        time = np.arange(signal_data.shape[0]) / fs
        
        # Plot each lead
        for i in range(n_leads):
            lead_name = header['lead_names'][i] if header and 'lead_names' in header else f"Lead {i}"
            axes[i].plot(time, signal_data[:, i])
            axes[i].set_title(lead_name)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True)
            
        # Hide unused subplots
        for i in range(n_leads, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create ECG preprocessor
    preprocessor = ECGPreprocessor(sampling_rate=500)
    
    # Example with synthetic data
    # Generate synthetic ECG-like signal
    t = np.linspace(0, 10, 5000)
    ecg = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*10*t) + 0.1*np.random.randn(len(t))
    
    # Add baseline wander
    baseline = 0.5*np.sin(2*np.pi*0.05*t)
    ecg_with_baseline = ecg + baseline
    
    # Preprocess signal
    processed_ecg = preprocessor.preprocess_ecg(ecg_with_baseline)
    
    # Detect R peaks
    r_peaks = preprocessor.detect_r_peaks(processed_ecg)
    
    # Plot original and processed signals
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_with_baseline)
    plt.title("Original ECG with Baseline Wander")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, processed_ecg)
    plt.title("Processed ECG")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, processed_ecg)
    plt.plot(t[r_peaks], processed_ecg[r_peaks], 'ro')
    plt.title("Processed ECG with Detected R Peaks")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

