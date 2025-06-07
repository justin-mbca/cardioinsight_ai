"""
CardioInsight AI - Main Program

This is the main program for the CardioInsight AI system, which integrates all modules
and provides a complete system interface.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import CardioInsight AI modules
from ecg_preprocessing import ECGPreprocessor
from feature_extraction import ECGFeatureExtractor as FeatureExtractor
from ml_models import ECGClassifier as MLModelManager
from dl_models import ECGDeepLearningModel as DLModelManager
from explainability import GradCAM as ExplainabilityModule
from multimodal_fusion import MultimodalFusion as MultimodalFusionModule
from dynamic_ecg_annotation import HolterAnalyzer
from teaching_module import TeachingSystem
from remote_healthcare import RemoteHealthcareSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cardioinsight.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CardioInsightAI:
    """
    Main class for the CardioInsight AI system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the CardioInsight AI system.
        
        Parameters:
        -----------
        config_path : str or None
            Path to the configuration file. If None, uses default configuration.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize modules
        self._initialize_modules()
        
        logger.info("CardioInsight AI system initialized")
        
    def _load_config(self, config_path):
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_path : str or None
            Path to the configuration file. If None, uses default configuration.
            
        Returns:
        --------
        config : dict
            Configuration dictionary.
        """
        # Default configuration
        default_config = {
            'data_dir': 'data',
            'models_dir': 'models',
            'results_dir': 'results',
            'use_gpu': True,
            'default_model': 'dl_model',
            'server_url': None,
            'api_key': None
        }
        
        if config_path is None or not os.path.exists(config_path):
            logger.info("Using default configuration")
            return default_config
            
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with default config
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
            
    def _initialize_modules(self):
        """
        Initialize all system modules.
        """
        try:
            # Create directories if they don't exist
            os.makedirs(self.config['data_dir'], exist_ok=True)
            os.makedirs(self.config['models_dir'], exist_ok=True)
            os.makedirs(self.config['results_dir'], exist_ok=True)
            
            # Initialize preprocessing module
            self.preprocessor = ECGPreprocessor()
            
            # Initialize feature extraction module
            self.feature_extractor = FeatureExtractor()
            
            # Initialize ML model manager
            self.ml_model_manager = MLModelManager(
                models_dir=os.path.join(self.config['models_dir'], 'ml_models')
            )
            
            # Initialize DL model manager
            self.dl_model_manager = DLModelManager(
                models_dir=os.path.join(self.config['models_dir'], 'dl_models'),
                use_gpu=self.config['use_gpu']
            )
            
            # Initialize explainability module
            self.explainability = ExplainabilityModule()
            
            # Initialize multimodal fusion module
            self.multimodal_fusion = MultimodalFusionModule()
            
            # Initialize Holter analyzer
            self.holter_analyzer = HolterAnalyzer()
            
            # Initialize teaching system
            self.teaching_system = TeachingSystem(
                case_library_path=os.path.join(self.config['data_dir'], 'case_library.json'),
                model_path=self._get_default_model_path()
            )
            
            # Initialize remote healthcare system
            self.remote_healthcare = RemoteHealthcareSystem(
                model_path=self._get_default_model_path(),
                server_url=self.config.get('server_url'),
                api_key=self.config.get('api_key')
            )
            
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            raise
            
    def _get_default_model_path(self):
        """
        Get the path to the default model.
        
        Returns:
        --------
        model_path : str or None
            Path to the default model, or None if not found.
        """
        if self.config['default_model'] == 'ml_model':
            model_dir = os.path.join(self.config['models_dir'], 'ml_models')
        else:
            model_dir = os.path.join(self.config['models_dir'], 'dl_models')
            
        if not os.path.exists(model_dir):
            return None
            
        # Find the latest model
        models = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.pkl')]
        
        if not models:
            return None
            
        # Sort by modification time (newest first)
        models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        
        return os.path.join(model_dir, models[0])
    
    def load_ecg_data(self, file_path):
        """
        Load ECG data from file.
        
        Parameters:
        -----------
        file_path : str
            Path to the ECG data file.
            
        Returns:
        --------
        ecg_data : array-like
            ECG data.
        metadata : dict
            Metadata.
        """
        try:
            # Check file extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.csv':
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Extract data and metadata
                if 'time' in df.columns:
                    metadata = {'sampling_rate': 1 / (df['time'][1] - df['time'][0])}
                    ecg_data = df.drop(columns=['time']).values
                else:
                    metadata = {}
                    ecg_data = df.values
                    
            elif ext.lower() == '.npy':
                # Load NumPy file
                ecg_data = np.load(file_path)
                metadata = {}
                
            elif ext.lower() == '.mat':
                # Load MATLAB file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                
                # Extract data and metadata
                # This depends on the structure of the MAT file
                # Here we assume the ECG data is stored in a variable named 'ecg'
                if 'ecg' in mat_data:
                    ecg_data = mat_data['ecg']
                else:
                    # Try to find the largest array
                    largest_var = None
                    largest_size = 0
                    
                    for var_name, var_value in mat_data.items():
                        if isinstance(var_value, np.ndarray) and var_value.size > largest_size:
                            largest_var = var_name
                            largest_size = var_value.size
                            
                    if largest_var is not None:
                        ecg_data = mat_data[largest_var]
                    else:
                        raise ValueError("Could not find ECG data in MAT file")
                        
                metadata = {}
                
                # Extract metadata if available
                if 'fs' in mat_data:
                    metadata['sampling_rate'] = float(mat_data['fs'])
                    
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            logger.info(f"ECG data loaded from {file_path}")
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading ECG data: {e}")
            raise
            
    def preprocess_ecg(self, ecg_data, sampling_rate=None):
        """
        Preprocess ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        sampling_rate : float or None
            Sampling rate in Hz. If None, uses default.
            
        Returns:
        --------
        processed_data : array-like
            Preprocessed ECG data.
        """
        try:
            # Set default sampling rate if not provided
            if sampling_rate is None:
                sampling_rate = 250  # Default sampling rate
                
            # Preprocess data
            processed_data = self.preprocessor.preprocess(ecg_data, sampling_rate)
            
            logger.info("ECG data preprocessed")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing ECG data: {e}")
            raise
            
    def extract_features(self, ecg_data, sampling_rate=None):
        """
        Extract features from ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        sampling_rate : float or None
            Sampling rate in Hz. If None, uses default.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        try:
            # Set default sampling rate if not provided
            if sampling_rate is None:
                sampling_rate = 250  # Default sampling rate
                
            # Extract features
            features = self.feature_extractor.extract_features(ecg_data, sampling_rate)
            
            logger.info("Features extracted from ECG data")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
            
    def analyze_ecg(self, ecg_data, metadata=None, use_dl=True, explain=True):
        """
        Analyze ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        metadata : dict or None
            Metadata. If None, uses empty dict.
        use_dl : bool
            Whether to use deep learning model. Default is True.
        explain : bool
            Whether to generate explanation. Default is True.
            
        Returns:
        --------
        results : dict
            Analysis results.
        """
        try:
            # Set metadata
            if metadata is None:
                metadata = {}
                
            # Get sampling rate
            sampling_rate = metadata.get('sampling_rate', 250)
            
            # Preprocess data
            processed_data = self.preprocess_ecg(ecg_data, sampling_rate)
            
            # Extract features
            features = self.extract_features(processed_data, sampling_rate)
            
            # Select model
            if use_dl:
                # Use deep learning model
                prediction, confidence = self.dl_model_manager.predict(processed_data)
                model_type = 'deep_learning'
            else:
                # Use machine learning model
                prediction, confidence = self.ml_model_manager.predict(features)
                model_type = 'machine_learning'
                
            # Generate explanation if requested
            explanation = None
            if explain:
                if use_dl:
                    explanation = self.explainability.explain_dl_prediction(
                        self.dl_model_manager.get_current_model(),
                        processed_data,
                        prediction
                    )
                else:
                    explanation = self.explainability.explain_ml_prediction(
                        self.ml_model_manager.get_current_model(),
                        features,
                        prediction
                    )
                    
            # Compile results
            results = {
                'prediction': prediction,
                'confidence': confidence,
                'model_type': model_type,
                'explanation': explanation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"ECG analysis complete. Prediction: {prediction}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing ECG: {e}")
            raise
            
    def analyze_with_multimodal(self, ecg_data, clinical_data, metadata=None):
        """
        Analyze ECG data with multimodal fusion.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        clinical_data : dict
            Clinical data (e.g., symptoms, medical history).
        metadata : dict or None
            Metadata. If None, uses empty dict.
            
        Returns:
        --------
        results : dict
            Analysis results.
        """
        try:
            # Set metadata
            if metadata is None:
                metadata = {}
                
            # Get sampling rate
            sampling_rate = metadata.get('sampling_rate', 250)
            
            # Preprocess ECG data
            processed_ecg = self.preprocess_ecg(ecg_data, sampling_rate)
            
            # Extract ECG features
            ecg_features = self.extract_features(processed_ecg, sampling_rate)
            
            # Process clinical data
            processed_clinical = self.multimodal_fusion.process_clinical_data(clinical_data)
            
            # Fuse data and make prediction
            prediction, confidence = self.multimodal_fusion.predict(ecg_features, processed_clinical)
            
            # Generate explanation
            explanation = self.explainability.explain_multimodal_prediction(
                self.multimodal_fusion.get_current_model(),
                ecg_features,
                processed_clinical,
                prediction
            )
            
            # Compile results
            results = {
                'prediction': prediction,
                'confidence': confidence,
                'model_type': 'multimodal',
                'explanation': explanation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Multimodal analysis complete. Prediction: {prediction}")
            return results
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            raise
            
    def analyze_holter(self, ecg_data, metadata=None):
        """
        Analyze Holter ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            Holter ECG data.
        metadata : dict or None
            Metadata. If None, uses empty dict.
            
        Returns:
        --------
        results : dict
            Analysis results.
        """
        try:
            # Set metadata
            if metadata is None:
                metadata = {}
                
            # Get sampling rate
            sampling_rate = metadata.get('sampling_rate', 250)
            
            # Analyze Holter data
            holter_results = self.holter_analyzer.analyze(ecg_data, sampling_rate)
            
            # Generate report
            report = self.holter_analyzer.generate_report(holter_results)
            
            # Compile results
            results = {
                'holter_analysis': holter_results,
                'report': report,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("Holter analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing Holter data: {e}")
            raise
            
    def create_teaching_quiz(self, n_questions=10, difficulty=None):
        """
        Create a teaching quiz.
        
        Parameters:
        -----------
        n_questions : int
            Number of questions. Default is 10.
        difficulty : str or None
            Difficulty level. If None, selects from any difficulty.
            
        Returns:
        --------
        quiz : dict
            Quiz data.
        """
        try:
            # Create quiz
            quiz = self.teaching_system.create_quiz(n_questions, difficulty)
            
            logger.info(f"Teaching quiz created with {n_questions} questions")
            return quiz
            
        except Exception as e:
            logger.error(f"Error creating teaching quiz: {e}")
            raise
            
    def process_for_remote(self, ecg_data, metadata=None):
        """
        Process ECG data for remote consultation.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        metadata : dict or None
            Metadata. If None, uses empty dict.
            
        Returns:
        --------
        result : dict
            Processing result.
        """
        try:
            # Set metadata
            if metadata is None:
                metadata = {}
                
            # Process ECG data
            result = self.remote_healthcare.process_ecg(ecg_data, metadata=metadata)
            
            logger.info(f"ECG processed for remote consultation. Case ID: {result.get('case_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing ECG for remote consultation: {e}")
            raise
            
    def optimize_for_remote(self):
        """
        Optimize models for remote deployment.
        
        Returns:
        --------
        result : dict
            Optimization result.
        """
        try:
            # Optimize model
            optimized_model = self.remote_healthcare.optimize_model()
            
            # Save optimized model
            model_path = os.path.join(self.config['models_dir'], 'optimized_model.h5')
            tflite_path = os.path.join(self.config['models_dir'], 'optimized_model.tflite')
            
            self.remote_healthcare.model_optimizer.save_optimized_model(model_path)
            self.remote_healthcare.model_optimizer.export_tflite_model(tflite_path)
            
            # Compile result
            result = {
                'model_path': model_path,
                'tflite_path': tflite_path,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("Model optimized for remote deployment")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing model for remote deployment: {e}")
            raise
            
    def save_results(self, results, filename=None):
        """
        Save analysis results.
        
        Parameters:
        -----------
        results : dict
            Analysis results.
        filename : str or None
            Filename to save results. If None, generates a filename.
            
        Returns:
        --------
        file_path : str
            Path to the saved results file.
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"results_{timestamp}.json"
                
            # Create file path
            file_path = os.path.join(self.config['results_dir'], filename)
            
            # Save results
            import json
            with open(file_path, 'w') as f:
                json.dump(results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                
            logger.info(f"Results saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    def load_results(self, file_path):
        """
        Load analysis results.
        
        Parameters:
        -----------
        file_path : str
            Path to the results file.
            
        Returns:
        --------
        results : dict
            Analysis results.
        """
        try:
            # Load results
            import json
            with open(file_path, 'r') as f:
                results = json.load(f)
                
            logger.info(f"Results loaded from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
            
    def shutdown(self):
        """
        Shutdown the system.
        """
        try:
            # Shutdown remote healthcare system
            self.remote_healthcare.shutdown()
            
            logger.info("CardioInsight AI system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise


def main():
    """
    Main function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CardioInsight AI - Multi-modal Intelligent ECG Analysis System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--ecg', type=str, help='Path to ECG data file')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--use-dl', action='store_true', help='Use deep learning model')
    parser.add_argument('--explain', action='store_true', help='Generate explanation')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for remote deployment')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = CardioInsightAI(args.config)
        
        # Check if optimization requested
        if args.optimize:
            result = system.optimize_for_remote()
            print(f"Model optimized for remote deployment. Saved to {result['model_path']}")
            return
            
        # Check if ECG file provided
        if args.ecg:
            # Load ECG data
            ecg_data, metadata = system.load_ecg_data(args.ecg)
            
            # Analyze ECG
            results = system.analyze_ecg(ecg_data, metadata, use_dl=args.use_dl, explain=args.explain)
            
            # Save results if output path provided
            if args.output:
                system.save_results(results, args.output)
                print(f"Results saved to {args.output}")
            else:
                # Print results
                print(f"Prediction: {results['prediction']}")
                print(f"Confidence: {results['confidence']}")
                
        else:
            print("No ECG file provided. Use --ecg to specify an ECG data file.")
            
        # Shutdown system
        system.shutdown()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

