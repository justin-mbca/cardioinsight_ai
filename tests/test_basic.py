"""
CardioInsight AI - Basic Tests

This script contains basic tests for the CardioInsight AI system.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path to import cardioinsight_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CardioInsight AI modules
from cardioinsight_ai.ecg_preprocessing import ECGPreprocessor
from cardioinsight_ai.feature_extraction import FeatureExtractor
from cardioinsight_ai.ml_models import MLModelManager
from cardioinsight_ai.dl_models import DLModelManager
from cardioinsight_ai.explainability import ExplainabilityModule
from cardioinsight_ai.main import CardioInsightAI

class TestECGPreprocessor(unittest.TestCase):
    """
    Test ECG preprocessing module.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.preprocessor = ECGPreprocessor()
        
        # Generate synthetic ECG data
        self.ecg_data = self.generate_synthetic_ecg()
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.preprocessor)
        
    def test_preprocess(self):
        """
        Test preprocess method.
        """
        # Preprocess data
        processed_data = self.preprocessor.preprocess(self.ecg_data, sampling_rate=250)
        
        # Check output shape
        self.assertEqual(processed_data.shape, self.ecg_data.shape)
        
        # Check that data has been normalized
        self.assertTrue(np.all(np.abs(processed_data) <= 1.0))
        
    def generate_synthetic_ecg(self, duration=10, sampling_rate=250, num_leads=12):
        """
        Generate synthetic ECG data for testing.
        
        Parameters:
        -----------
        duration : float
            Duration in seconds. Default is 10.
        sampling_rate : int
            Sampling rate in Hz. Default is 250.
        num_leads : int
            Number of leads. Default is 12.
            
        Returns:
        --------
        ecg_data : array-like
            Synthetic ECG data.
        """
        # Number of samples
        n_samples = int(duration * sampling_rate)
        
        # Time array
        t = np.arange(n_samples) / sampling_rate
        
        # Generate synthetic ECG for each lead
        ecg_data = np.zeros((n_samples, num_leads))
        
        for lead in range(num_leads):
            # Base frequency and phase
            base_freq = 1.2  # ~72 bpm
            phase = lead * 0.2
            
            # QRS complex
            qrs = np.sin(2 * np.pi * base_freq * t + phase) * np.exp(-((t % (1/base_freq) - 0.1) ** 2) / 0.001)
            
            # P wave
            p_wave = 0.2 * np.sin(2 * np.pi * base_freq * t + phase - 0.2) * np.exp(-((t % (1/base_freq) - 0.05) ** 2) / 0.002)
            
            # T wave
            t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
            
            # Combine waves
            ecg = qrs + p_wave + t_wave
            
            # Add noise
            noise = 0.05 * np.random.randn(n_samples)
            
            # Add baseline wander
            baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t + lead)
            
            # Combine all components
            ecg_data[:, lead] = ecg + noise + baseline
            
        return ecg_data

class TestFeatureExtractor(unittest.TestCase):
    """
    Test feature extraction module.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.feature_extractor = FeatureExtractor()
        
        # Generate synthetic ECG data
        self.ecg_data = self.generate_synthetic_ecg()
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.feature_extractor)
        
    def test_extract_features(self):
        """
        Test extract_features method.
        """
        # Extract features
        features = self.feature_extractor.extract_features(self.ecg_data, sampling_rate=250)
        
        # Check that features are not None
        self.assertIsNotNone(features)
        
        # Check that features have the expected shape
        self.assertEqual(features.shape[0], self.ecg_data.shape[1])  # One feature vector per lead
        
    def generate_synthetic_ecg(self, duration=10, sampling_rate=250, num_leads=12):
        """
        Generate synthetic ECG data for testing.
        """
        # Number of samples
        n_samples = int(duration * sampling_rate)
        
        # Time array
        t = np.arange(n_samples) / sampling_rate
        
        # Generate synthetic ECG for each lead
        ecg_data = np.zeros((n_samples, num_leads))
        
        for lead in range(num_leads):
            # Base frequency and phase
            base_freq = 1.2  # ~72 bpm
            phase = lead * 0.2
            
            # QRS complex
            qrs = np.sin(2 * np.pi * base_freq * t + phase) * np.exp(-((t % (1/base_freq) - 0.1) ** 2) / 0.001)
            
            # P wave
            p_wave = 0.2 * np.sin(2 * np.pi * base_freq * t + phase - 0.2) * np.exp(-((t % (1/base_freq) - 0.05) ** 2) / 0.002)
            
            # T wave
            t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
            
            # Combine waves
            ecg = qrs + p_wave + t_wave
            
            # Add noise
            noise = 0.05 * np.random.randn(n_samples)
            
            # Add baseline wander
            baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t + lead)
            
            # Combine all components
            ecg_data[:, lead] = ecg + noise + baseline
            
        return ecg_data

class TestMLModelManager(unittest.TestCase):
    """
    Test ML model manager module.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.ml_model_manager = MLModelManager(models_dir='models/ml_models')
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.ml_model_manager)
        
    def test_get_available_models(self):
        """
        Test get_available_models method.
        """
        # Get available models
        models = self.ml_model_manager.get_available_models()
        
        # Check that models is a list
        self.assertIsInstance(models, list)

class TestDLModelManager(unittest.TestCase):
    """
    Test DL model manager module.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.dl_model_manager = DLModelManager(models_dir='models/dl_models', use_gpu=False)
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.dl_model_manager)
        
    def test_get_available_models(self):
        """
        Test get_available_models method.
        """
        # Get available models
        models = self.dl_model_manager.get_available_models()
        
        # Check that models is a list
        self.assertIsInstance(models, list)

class TestExplainabilityModule(unittest.TestCase):
    """
    Test explainability module.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.explainability = ExplainabilityModule()
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.explainability)

class TestCardioInsightAI(unittest.TestCase):
    """
    Test CardioInsight AI main class.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        self.system = CardioInsightAI()
        
    def test_initialization(self):
        """
        Test initialization.
        """
        self.assertIsNotNone(self.system)
        
    def test_load_ecg_data(self):
        """
        Test load_ecg_data method with synthetic data.
        """
        # Generate synthetic ECG data
        ecg_data = self.generate_synthetic_ecg()
        
        # Save to CSV file
        import pandas as pd
        df = pd.DataFrame(ecg_data)
        csv_path = 'test_ecg.csv'
        df.to_csv(csv_path, index=False)
        
        try:
            # Load ECG data
            loaded_data, metadata = self.system.load_ecg_data(csv_path)
            
            # Check that data is loaded correctly
            self.assertEqual(loaded_data.shape, ecg_data.shape)
            
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.remove(csv_path)
                
    def generate_synthetic_ecg(self, duration=10, sampling_rate=250, num_leads=12):
        """
        Generate synthetic ECG data for testing.
        """
        # Number of samples
        n_samples = int(duration * sampling_rate)
        
        # Time array
        t = np.arange(n_samples) / sampling_rate
        
        # Generate synthetic ECG for each lead
        ecg_data = np.zeros((n_samples, num_leads))
        
        for lead in range(num_leads):
            # Base frequency and phase
            base_freq = 1.2  # ~72 bpm
            phase = lead * 0.2
            
            # QRS complex
            qrs = np.sin(2 * np.pi * base_freq * t + phase) * np.exp(-((t % (1/base_freq) - 0.1) ** 2) / 0.001)
            
            # P wave
            p_wave = 0.2 * np.sin(2 * np.pi * base_freq * t + phase - 0.2) * np.exp(-((t % (1/base_freq) - 0.05) ** 2) / 0.002)
            
            # T wave
            t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
            
            # Combine waves
            ecg = qrs + p_wave + t_wave
            
            # Add noise
            noise = 0.05 * np.random.randn(n_samples)
            
            # Add baseline wander
            baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t + lead)
            
            # Combine all components
            ecg_data[:, lead] = ecg + noise + baseline
            
        return ecg_data

if __name__ == '__main__':
    unittest.main()

