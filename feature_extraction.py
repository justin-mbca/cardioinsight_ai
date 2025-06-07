"""
CardioInsight AI - Feature Extraction Module

This module provides functions for extracting features from preprocessed ECG data.
It includes time domain, frequency domain, and wavelet-based feature extraction methods.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ECGFeatureExtractor:
    """
    Class for extracting features from ECG signals.
    """
    
    def __init__(self, sampling_rate=500):
        """
        Initialize the ECG feature extractor.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the ECG data in Hz. Default is 500 Hz.
        """
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, ecg_segment):
        """
        Extract time domain features from an ECG segment.
        
        Parameters:
        -----------
        ecg_segment : ndarray
            ECG segment with shape (n_samples,) or (n_samples, n_leads).
            
        Returns:
        --------
        features : dict
            Dictionary of time domain features.
        """
        # Handle multi-lead data
        if ecg_segment.ndim > 1:
            # Extract features for each lead and average
            n_leads = ecg_segment.shape[1]
            all_features = []
            
            for i in range(n_leads):
                lead_features = self.extract_time_domain_features(ecg_segment[:, i])
                all_features.append(lead_features)
                
            # Average features across leads
            features = {}
            for key in all_features[0].keys():
                features[key] = np.mean([f[key] for f in all_features])
                
            return features
        
        # Single lead processing
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(ecg_segment)
        features['std'] = np.std(ecg_segment)
        features['var'] = np.var(ecg_segment)
        features['skewness'] = stats.skew(ecg_segment)
        features['kurtosis'] = stats.kurtosis(ecg_segment)
        features['rms'] = np.sqrt(np.mean(ecg_segment**2))
        
        # Range-based features
        features['range'] = np.max(ecg_segment) - np.min(ecg_segment)
        features['peak_to_peak'] = features['range']
        features['abs_max'] = np.max(np.abs(ecg_segment))
        
        # Percentile-based features
        features['percentile_25'] = np.percentile(ecg_segment, 25)
        features['percentile_50'] = np.percentile(ecg_segment, 50)  # median
        features['percentile_75'] = np.percentile(ecg_segment, 75)
        features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(ecg_segment)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_segment)
        
        # Energy and power
        features['energy'] = np.sum(ecg_segment**2)
        features['power'] = features['energy'] / len(ecg_segment)
        
        # First and second derivatives
        first_derivative = np.diff(ecg_segment)
        second_derivative = np.diff(first_derivative)
        
        features['mean_abs_derivative'] = np.mean(np.abs(first_derivative))
        features['max_derivative'] = np.max(np.abs(first_derivative))
        features['mean_abs_second_derivative'] = np.mean(np.abs(second_derivative))
        features['max_second_derivative'] = np.max(np.abs(second_derivative))
        
        return features
    
    def extract_frequency_domain_features(self, ecg_segment, nperseg=None):
        """
        Extract frequency domain features from an ECG segment.
        
        Parameters:
        -----------
        ecg_segment : ndarray
            ECG segment with shape (n_samples,) or (n_samples, n_leads).
        nperseg : int or None
            Length of each segment for FFT. If None, uses min(256, len(ecg_segment)).
            
        Returns:
        --------
        features : dict
            Dictionary of frequency domain features.
        """
        # Handle multi-lead data
        if ecg_segment.ndim > 1:
            # Extract features for each lead and average
            n_leads = ecg_segment.shape[1]
            all_features = []
            
            for i in range(n_leads):
                lead_features = self.extract_frequency_domain_features(ecg_segment[:, i], nperseg)
                all_features.append(lead_features)
                
            # Average features across leads
            features = {}
            for key in all_features[0].keys():
                features[key] = np.mean([f[key] for f in all_features])
                
            return features
        
        # Single lead processing
        features = {}
        
        # Set default nperseg if not provided
        if nperseg is None:
            nperseg = min(256, len(ecg_segment))
        
        # Compute power spectral density
        frequencies, psd = signal.welch(ecg_segment, fs=self.sampling_rate, nperseg=nperseg)
        
        # Total power
        features['total_power'] = np.sum(psd)
        
        # Power in different frequency bands
        # VLF: 0-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
        vlf_mask = (frequencies >= 0) & (frequencies < 0.04)
        lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)
        
        features['vlf_power'] = np.sum(psd[vlf_mask])
        features['lf_power'] = np.sum(psd[lf_mask])
        features['hf_power'] = np.sum(psd[hf_mask])
        
        # Power ratios
        if features['lf_power'] > 0:
            features['lf_hf_ratio'] = features['hf_power'] / features['lf_power']
        else:
            features['lf_hf_ratio'] = 0
            
        if features['total_power'] > 0:
            features['vlf_power_ratio'] = features['vlf_power'] / features['total_power']
            features['lf_power_ratio'] = features['lf_power'] / features['total_power']
            features['hf_power_ratio'] = features['hf_power'] / features['total_power']
        else:
            features['vlf_power_ratio'] = 0
            features['lf_power_ratio'] = 0
            features['hf_power_ratio'] = 0
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Avoid log(0)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        
        # Dominant frequency
        features['dominant_frequency'] = frequencies[np.argmax(psd)]
        
        # Spectral edge frequency (95% of power)
        total_power = np.sum(psd)
        power_sum = 0
        for i, p in enumerate(psd):
            power_sum += p
            if power_sum >= 0.95 * total_power:
                features['spectral_edge_frequency'] = frequencies[i]
                break
        else:
            features['spectral_edge_frequency'] = frequencies[-1]
        
        # Spectral moments
        features['spectral_mean'] = np.sum(frequencies * psd) / np.sum(psd)
        features['spectral_std'] = np.sqrt(np.sum((frequencies - features['spectral_mean'])**2 * psd) / np.sum(psd))
        features['spectral_skewness'] = np.sum((frequencies - features['spectral_mean'])**3 * psd) / (np.sum(psd) * features['spectral_std']**3)
        features['spectral_kurtosis'] = np.sum((frequencies - features['spectral_mean'])**4 * psd) / (np.sum(psd) * features['spectral_std']**4) - 3
        
        return features
    
    def extract_wavelet_features(self, ecg_segment, wavelet='db4', level=5):
        """
        Extract wavelet-based features from an ECG segment.
        
        Parameters:
        -----------
        ecg_segment : ndarray
            ECG segment with shape (n_samples,) or (n_samples, n_leads).
        wavelet : str
            Wavelet type. Default is 'db4'.
        level : int
            Decomposition level. Default is 5.
            
        Returns:
        --------
        features : dict
            Dictionary of wavelet-based features.
        """
        # Handle multi-lead data
        if ecg_segment.ndim > 1:
            # Extract features for each lead and average
            n_leads = ecg_segment.shape[1]
            all_features = []
            
            for i in range(n_leads):
                lead_features = self.extract_wavelet_features(ecg_segment[:, i], wavelet, level)
                all_features.append(lead_features)
                
            # Average features across leads
            features = {}
            for key in all_features[0].keys():
                features[key] = np.mean([f[key] for f in all_features])
                
            return features
        
        # Single lead processing
        features = {}
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(ecg_segment, wavelet, level=level)
        
        # Extract features from each level
        for i, coef in enumerate(coeffs):
            if i == 0:
                prefix = 'a'  # Approximation coefficients
            else:
                prefix = f'd{level-i+1}'  # Detail coefficients
                
            # Statistical features
            features[f'{prefix}_mean'] = np.mean(coef)
            features[f'{prefix}_std'] = np.std(coef)
            features[f'{prefix}_energy'] = np.sum(coef**2)
            features[f'{prefix}_entropy'] = -np.sum(coef**2 * np.log2(coef**2 + 1e-10))
            
        # Relative energy features
        total_energy = sum(features[f'{"a" if i == 0 else f"d{level-i+1}"}_energy'] for i in range(len(coeffs)))
        
        if total_energy > 0:
            for i in range(len(coeffs)):
                if i == 0:
                    prefix = 'a'
                else:
                    prefix = f'd{level-i+1}'
                    
                features[f'{prefix}_rel_energy'] = features[f'{prefix}_energy'] / total_energy
        
        return features
    
    def extract_heartbeat_features(self, heartbeat, r_peak_idx=None):
        """
        Extract features from a single heartbeat.
        
        Parameters:
        -----------
        heartbeat : ndarray
            Heartbeat signal with shape (n_samples,) or (n_samples, n_leads).
        r_peak_idx : int or None
            Index of R peak in the heartbeat. If None, tries to detect it.
            
        Returns:
        --------
        features : dict
            Dictionary of heartbeat features.
        """
        # Handle multi-lead data
        if heartbeat.ndim > 1:
            # Extract features for each lead and combine
            n_leads = heartbeat.shape[1]
            all_features = []
            
            for i in range(n_leads):
                lead_features = self.extract_heartbeat_features(heartbeat[:, i], r_peak_idx)
                # Add lead prefix to feature names
                lead_features = {f'lead{i}_{key}': value for key, value in lead_features.items()}
                all_features.append(lead_features)
                
            # Combine features from all leads
            features = {}
            for f in all_features:
                features.update(f)
                
            return features
        
        # Single lead processing
        features = {}
        
        # Detect R peak if not provided
        if r_peak_idx is None:
            r_peak_idx = np.argmax(heartbeat)
            
        # Basic heartbeat features
        features['r_peak_amplitude'] = heartbeat[r_peak_idx]
        features['heartbeat_mean'] = np.mean(heartbeat)
        features['heartbeat_std'] = np.std(heartbeat)
        
        # Pre-R and post-R segment features
        pre_r = heartbeat[:r_peak_idx]
        post_r = heartbeat[r_peak_idx:]
        
        if len(pre_r) > 0:
            features['pre_r_mean'] = np.mean(pre_r)
            features['pre_r_std'] = np.std(pre_r)
            features['pre_r_min'] = np.min(pre_r)
            features['pre_r_max'] = np.max(pre_r)
            
            # Q wave (minimum before R peak)
            q_idx = np.argmin(pre_r[-min(len(pre_r), 50):])  # Look at last 50 samples before R
            q_idx = len(pre_r) - min(len(pre_r), 50) + q_idx
            features['q_amplitude'] = pre_r[q_idx]
            features['q_r_diff'] = features['r_peak_amplitude'] - features['q_amplitude']
            
        if len(post_r) > 0:
            features['post_r_mean'] = np.mean(post_r)
            features['post_r_std'] = np.std(post_r)
            features['post_r_min'] = np.min(post_r)
            features['post_r_max'] = np.max(post_r)
            
            # S wave (minimum after R peak)
            s_idx = np.argmin(post_r[:min(len(post_r), 50)])  # Look at first 50 samples after R
            features['s_amplitude'] = post_r[s_idx]
            features['r_s_diff'] = features['r_peak_amplitude'] - features['s_amplitude']
            
            # T wave (maximum after S wave)
            if s_idx + 1 < len(post_r):
                t_segment = post_r[s_idx+1:min(len(post_r), s_idx+150)]  # Look up to 150 samples after S
                if len(t_segment) > 0:
                    t_idx = np.argmax(t_segment) + s_idx + 1
                    features['t_amplitude'] = post_r[t_idx]
                    features['s_t_diff'] = features['t_amplitude'] - features['s_amplitude']
        
        return features
    
    def extract_rhythm_features(self, r_peaks, signal_length=None):
        """
        Extract rhythm-based features from R peak locations.
        
        Parameters:
        -----------
        r_peaks : ndarray
            Indices of R peaks.
        signal_length : int or None
            Length of the original signal. If None, uses max(r_peaks) + 1.
            
        Returns:
        --------
        features : dict
            Dictionary of rhythm features.
        """
        features = {}
        
        if len(r_peaks) < 2:
            # Not enough R peaks for rhythm analysis
            features['mean_rr'] = 0
            features['std_rr'] = 0
            features['rmssd'] = 0
            features['pnn50'] = 0
            features['heart_rate'] = 0
            features['heart_rate_std'] = 0
            return features
        
        # Calculate RR intervals (in samples)
        rr_intervals = np.diff(r_peaks)
        
        # Convert to time (seconds)
        rr_intervals_sec = rr_intervals / self.sampling_rate
        
        # Basic RR interval features
        features['mean_rr'] = np.mean(rr_intervals_sec)
        features['std_rr'] = np.std(rr_intervals_sec)
        features['min_rr'] = np.min(rr_intervals_sec)
        features['max_rr'] = np.max(rr_intervals_sec)
        features['range_rr'] = features['max_rr'] - features['min_rr']
        
        # Heart rate features
        hr = 60 / rr_intervals_sec  # beats per minute
        features['heart_rate'] = np.mean(hr)
        features['heart_rate_std'] = np.std(hr)
        features['heart_rate_min'] = np.min(hr)
        features['heart_rate_max'] = np.max(hr)
        features['heart_rate_range'] = features['heart_rate_max'] - features['heart_rate_min']
        
        # Heart rate variability features
        # RMSSD: Root Mean Square of Successive Differences
        rr_diffs = np.diff(rr_intervals_sec)
        features['rmssd'] = np.sqrt(np.mean(rr_diffs**2))
        
        # pNN50: Percentage of successive RR intervals that differ by more than 50 ms
        nn50 = np.sum(np.abs(rr_diffs) > 0.05)  # 50 ms = 0.05 s
        if len(rr_diffs) > 0:
            features['pnn50'] = nn50 / len(rr_diffs) * 100
        else:
            features['pnn50'] = 0
            
        # SDSD: Standard Deviation of Successive Differences
        features['sdsd'] = np.std(rr_diffs)
        
        # Triangular index
        if signal_length is None:
            signal_length = np.max(r_peaks) + 1
            
        features['triangular_index'] = len(r_peaks) / signal_length * self.sampling_rate
        
        # Poincaré plot features
        if len(rr_intervals_sec) > 1:
            rr_n = rr_intervals_sec[:-1]
            rr_n1 = rr_intervals_sec[1:]
            
            # SD1: Standard deviation perpendicular to the line of identity
            sd1 = np.std((rr_n1 - rr_n) / np.sqrt(2))
            features['sd1'] = sd1
            
            # SD2: Standard deviation along the line of identity
            sd2 = np.std((rr_n1 + rr_n) / np.sqrt(2))
            features['sd2'] = sd2
            
            # SD1/SD2 ratio
            if sd2 > 0:
                features['sd1_sd2_ratio'] = sd1 / sd2
            else:
                features['sd1_sd2_ratio'] = 0
        
        return features
    
    def extract_all_features(self, ecg_data, r_peaks=None):
        """
        Extract all features from ECG data.
        
        Parameters:
        -----------
        ecg_data : ndarray
            ECG data with shape (n_samples,) or (n_samples, n_leads).
        r_peaks : ndarray or None
            Indices of R peaks. If None, tries to detect them.
            
        Returns:
        --------
        features : dict
            Dictionary of all extracted features.
        """
        features = {}
        
        # Extract time domain features
        time_features = self.extract_time_domain_features(ecg_data)
        features.update({f'time_{key}': value for key, value in time_features.items()})
        
        # Extract frequency domain features
        freq_features = self.extract_frequency_domain_features(ecg_data)
        features.update({f'freq_{key}': value for key, value in freq_features.items()})
        
        # Extract wavelet features
        wavelet_features = self.extract_wavelet_features(ecg_data)
        features.update({f'wavelet_{key}': value for key, value in wavelet_features.items()})
        
        # Detect R peaks if not provided
        if r_peaks is None:
            from ecg_preprocessing import ECGPreprocessor
            preprocessor = ECGPreprocessor(sampling_rate=self.sampling_rate)
            
            # Use lead 0 for R peak detection if multi-lead
            if ecg_data.ndim > 1:
                r_peaks = preprocessor.detect_r_peaks(ecg_data[:, 0])
            else:
                r_peaks = preprocessor.detect_r_peaks(ecg_data)
        
        # Extract rhythm features
        if len(r_peaks) > 0:
            rhythm_features = self.extract_rhythm_features(r_peaks, len(ecg_data))
            features.update({f'rhythm_{key}': value for key, value in rhythm_features.items()})
            
            # Extract heartbeat features (average across all heartbeats)
            if ecg_data.ndim > 1:
                # Multi-lead data
                n_leads = ecg_data.shape[1]
                all_heartbeat_features = []
                
                for i in range(n_leads):
                    lead_heartbeat_features = []
                    
                    for r_idx in r_peaks:
                        # Define window boundaries
                        start = max(0, r_idx - 100)
                        end = min(len(ecg_data), r_idx + 100)
                        
                        # Extract heartbeat
                        heartbeat = ecg_data[start:end, i]
                        r_peak_idx = r_idx - start
                        
                        # Extract features
                        hb_features = self.extract_heartbeat_features(heartbeat, r_peak_idx)
                        lead_heartbeat_features.append(hb_features)
                    
                    # Average features across heartbeats
                    if lead_heartbeat_features:
                        avg_features = {}
                        for key in lead_heartbeat_features[0].keys():
                            avg_features[key] = np.mean([hb[key] for hb in lead_heartbeat_features if key in hb])
                        
                        # Add lead prefix
                        avg_features = {f'lead{i}_{key}': value for key, value in avg_features.items()}
                        all_heartbeat_features.append(avg_features)
                
                # Combine features from all leads
                heartbeat_features = {}
                for f in all_heartbeat_features:
                    heartbeat_features.update(f)
            else:
                # Single lead data
                heartbeat_features_list = []
                
                for r_idx in r_peaks:
                    # Define window boundaries
                    start = max(0, r_idx - 100)
                    end = min(len(ecg_data), r_idx + 100)
                    
                    # Extract heartbeat
                    heartbeat = ecg_data[start:end]
                    r_peak_idx = r_idx - start
                    
                    # Extract features
                    hb_features = self.extract_heartbeat_features(heartbeat, r_peak_idx)
                    heartbeat_features_list.append(hb_features)
                
                # Average features across heartbeats
                heartbeat_features = {}
                if heartbeat_features_list:
                    for key in heartbeat_features_list[0].keys():
                        heartbeat_features[key] = np.mean([hb[key] for hb in heartbeat_features_list if key in hb])
            
            features.update({f'beat_{key}': value for key, value in heartbeat_features.items()})
        
        return features
    
    def extract_features_from_segments(self, segments, r_peaks_list=None):
        """
        Extract features from multiple ECG segments.
        
        Parameters:
        -----------
        segments : list or ndarray
            List of ECG segments or 3D array with shape (n_segments, n_samples, n_leads).
        r_peaks_list : list or None
            List of R peak indices for each segment. If None, tries to detect them.
            
        Returns:
        --------
        features_df : DataFrame
            DataFrame of extracted features for all segments.
        """
        all_features = []
        
        for i, segment in enumerate(segments):
            # Get R peaks for this segment
            r_peaks = None
            if r_peaks_list is not None and i < len(r_peaks_list):
                r_peaks = r_peaks_list[i]
                
            # Extract features
            features = self.extract_all_features(segment, r_peaks)
            all_features.append(features)
            
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        return features_df
    
    def normalize_features(self, features_df, scaler=None):
        """
        Normalize features using StandardScaler.
        
        Parameters:
        -----------
        features_df : DataFrame
            DataFrame of extracted features.
        scaler : StandardScaler or None
            Scaler to use. If None, creates a new scaler.
            
        Returns:
        --------
        normalized_df : DataFrame
            DataFrame of normalized features.
        scaler : StandardScaler
            Fitted scaler.
        """
        # Create scaler if not provided
        if scaler is None:
            scaler = StandardScaler()
            
        # Fit and transform
        normalized_data = scaler.fit_transform(features_df)
        normalized_df = pd.DataFrame(normalized_data, columns=features_df.columns)
        
        return normalized_df, scaler
    
    def select_features(self, features_df, n_features=20, method='variance'):
        """
        Select most important features.
        
        Parameters:
        -----------
        features_df : DataFrame
            DataFrame of extracted features.
        n_features : int
            Number of features to select. Default is 20.
        method : str
            Feature selection method. Options: 'variance', 'correlation'. Default is 'variance'.
            
        Returns:
        --------
        selected_df : DataFrame
            DataFrame of selected features.
        """
        if method == 'variance':
            # Select features with highest variance
            variances = features_df.var().sort_values(ascending=False)
            selected_features = variances.index[:n_features]
            
        elif method == 'correlation':
            # Select features with lowest correlation to others
            corr_matrix = features_df.corr().abs()
            
            # Calculate mean correlation for each feature
            mean_corr = corr_matrix.mean().sort_values()
            selected_features = mean_corr.index[:n_features]
            
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
            
        # Return selected features
        return features_df[selected_features]
    
    def plot_feature_importance(self, features_df, n_features=20):
        """
        Plot feature importance based on variance.
        
        Parameters:
        -----------
        features_df : DataFrame
            DataFrame of extracted features.
        n_features : int
            Number of features to show. Default is 20.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Calculate variance for each feature
        variances = features_df.var().sort_values(ascending=False)
        top_features = variances.index[:n_features]
        top_variances = variances.values[:n_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(n_features), top_variances)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(top_features, rotation=90)
        ax.set_title('Feature Importance (Variance)')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Variance')
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Import ECG preprocessor
    from ecg_preprocessing import ECGPreprocessor
    
    # Create ECG preprocessor and feature extractor
    preprocessor = ECGPreprocessor(sampling_rate=500)
    feature_extractor = ECGFeatureExtractor(sampling_rate=500)
    
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
    
    # Extract features
    features = feature_extractor.extract_all_features(processed_ecg, r_peaks)
    
    # Print some features
    print("Number of features extracted:", len(features))
    print("\nTime domain features:")
    for key, value in {k: v for k, v in features.items() if k.startswith('time_')}.items():
        print(f"{key}: {value}")
        
    print("\nFrequency domain features:")
    for key, value in {k: v for k, v in features.items() if k.startswith('freq_')}.items():
        print(f"{key}: {value}")
        
    print("\nRhythm features:")
    for key, value in {k: v for k, v in features.items() if k.startswith('rhythm_')}.items():
        print(f"{key}: {value}")
        
    # Create segments for feature extraction
    segments = []
    for i in range(0, len(processed_ecg) - 1000, 500):  # 1000-sample segments with 500-sample overlap
        segments.append(processed_ecg[i:i+1000])
        
    # Extract features from segments
    features_df = feature_extractor.extract_features_from_segments(segments)
    
    # Normalize features
    normalized_df, scaler = feature_extractor.normalize_features(features_df)
    
    # Select top features
    selected_df = feature_extractor.select_features(normalized_df, n_features=10)
    
    print("\nSelected features:")
    print(selected_df.head())

