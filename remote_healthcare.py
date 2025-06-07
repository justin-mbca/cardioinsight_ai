"""
CardioInsight AI - Remote Healthcare Module

This module provides tools for adapting the CardioInsight AI system for remote and rural healthcare settings.
It includes model optimization for low-resource environments, offline processing capabilities,
and remote consultation features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import threading
import queue
import logging
import base64
import io
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Class for optimizing AI models for low-resource environments.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the model optimizer.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the model to optimize. If None, no model is loaded.
        """
        self.original_model = None
        self.optimized_model = None
        
        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """
        Load a model.
        
        Parameters:
        -----------
        model_path : str
            Path to the model.
        """
        try:
            self.original_model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def quantize_model(self, quantization_type='dynamic'):
        """
        Quantize the model to reduce size and improve inference speed.
        
        Parameters:
        -----------
        quantization_type : str
            Type of quantization. Options: 'dynamic', 'float16', 'int8'. Default is 'dynamic'.
            
        Returns:
        --------
        optimized_model : Model
            Quantized model.
        """
        if self.original_model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        try:
            if quantization_type == 'dynamic':
                # Dynamic range quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                # Convert back to TF model (for demonstration)
                # In practice, you would use the TFLite model directly
                self.optimized_model = self._convert_tflite_to_keras(tflite_model)
                
            elif quantization_type == 'float16':
                # Float16 quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                tflite_model = converter.convert()
                
                # Convert back to TF model (for demonstration)
                self.optimized_model = self._convert_tflite_to_keras(tflite_model)
                
            elif quantization_type == 'int8':
                # Int8 quantization (requires representative dataset)
                # This is a simplified example; in practice, you need a representative dataset
                def representative_dataset():
                    for _ in range(100):
                        yield [np.random.randn(1, *self.original_model.input_shape[1:]).astype(np.float32)]
                        
                converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                tflite_model = converter.convert()
                
                # Convert back to TF model (for demonstration)
                self.optimized_model = self._convert_tflite_to_keras(tflite_model)
                
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
                
            logger.info(f"Model quantized using {quantization_type} quantization")
            return self.optimized_model
            
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            raise
            
    def _convert_tflite_to_keras(self, tflite_model):
        """
        Convert TFLite model back to Keras model (for demonstration purposes).
        
        Parameters:
        -----------
        tflite_model : bytes
            TFLite model.
            
        Returns:
        --------
        keras_model : Model
            Keras model.
        """
        # In practice, you would use the TFLite model directly
        # This is a placeholder function
        return self.original_model
    
    def prune_model(self, target_sparsity=0.5):
        """
        Prune the model to reduce size and improve inference speed.
        
        Parameters:
        -----------
        target_sparsity : float
            Target sparsity (fraction of weights to prune). Default is 0.5.
            
        Returns:
        --------
        pruned_model : Model
            Pruned model.
        """
        if self.original_model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        try:
            # Apply pruning to the model
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.original_model,
                **pruning_params
            )
            
            # Compile the pruned model
            pruned_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fake training to apply pruning
            # In practice, you would train on real data
            fake_x = np.random.randn(10, *self.original_model.input_shape[1:])
            fake_y = np.random.randint(0, 2, (10, self.original_model.output_shape[1]))
            
            pruned_model.fit(
                fake_x, fake_y,
                epochs=1,
                callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
            )
            
            # Strip pruning wrapper
            self.optimized_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
            
            logger.info(f"Model pruned with target sparsity {target_sparsity}")
            return self.optimized_model
            
        except Exception as e:
            logger.error(f"Error pruning model: {e}")
            raise
            
    def evaluate_model(self, model, x_test, y_test):
        """
        Evaluate a model.
        
        Parameters:
        -----------
        model : Model
            Model to evaluate.
        x_test : array-like
            Test data.
        y_test : array-like
            Test labels.
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics.
        """
        try:
            # Predict
            y_pred = model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
            
            # Measure inference time
            start_time = time.time()
            model.predict(x_test[:10])
            end_time = time.time()
            inference_time = (end_time - start_time) / 10
            
            # Get model size
            model_size = self._get_model_size(model)
            
            # Compile metrics
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'inference_time': inference_time,
                'model_size': model_size
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
            
    def _get_model_size(self, model):
        """
        Get the size of a model in MB.
        
        Parameters:
        -----------
        model : Model
            Model to get size of.
            
        Returns:
        --------
        size : float
            Model size in MB.
        """
        # Save model to temporary file
        temp_path = 'temp_model.h5'
        model.save(temp_path)
        
        # Get file size
        size = os.path.getsize(temp_path) / (1024 * 1024)  # Convert to MB
        
        # Remove temporary file
        os.remove(temp_path)
        
        return size
    
    def compare_models(self, x_test, y_test):
        """
        Compare original and optimized models.
        
        Parameters:
        -----------
        x_test : array-like
            Test data.
        y_test : array-like
            Test labels.
            
        Returns:
        --------
        comparison : dict
            Dictionary containing comparison results.
        """
        if self.original_model is None:
            raise ValueError("No original model loaded.")
            
        if self.optimized_model is None:
            raise ValueError("No optimized model created. Call quantize_model() or prune_model() first.")
            
        try:
            # Evaluate original model
            original_metrics = self.evaluate_model(self.original_model, x_test, y_test)
            
            # Evaluate optimized model
            optimized_metrics = self.evaluate_model(self.optimized_model, x_test, y_test)
            
            # Calculate differences
            accuracy_diff = optimized_metrics['accuracy'] - original_metrics['accuracy']
            inference_time_diff = original_metrics['inference_time'] - optimized_metrics['inference_time']
            size_diff = original_metrics['model_size'] - optimized_metrics['model_size']
            
            # Compile comparison
            comparison = {
                'original': original_metrics,
                'optimized': optimized_metrics,
                'accuracy_diff': accuracy_diff,
                'inference_time_diff': inference_time_diff,
                'size_diff': size_diff
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
            
    def save_optimized_model(self, path):
        """
        Save the optimized model.
        
        Parameters:
        -----------
        path : str
            Path to save the model.
        """
        if self.optimized_model is None:
            raise ValueError("No optimized model created. Call quantize_model() or prune_model() first.")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            self.optimized_model.save(path)
            logger.info(f"Optimized model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving optimized model: {e}")
            raise
            
    def export_tflite_model(self, path):
        """
        Export the model to TFLite format.
        
        Parameters:
        -----------
        path : str
            Path to save the TFLite model.
        """
        if self.optimized_model is None:
            model = self.original_model
        else:
            model = self.optimized_model
            
        if model is None:
            raise ValueError("No model loaded.")
            
        try:
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save TFLite model
            with open(path, 'wb') as f:
                f.write(tflite_model)
                
            logger.info(f"TFLite model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error exporting TFLite model: {e}")
            raise


class OfflineProcessor:
    """
    Class for offline processing of ECG data.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the offline processor.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the model. If None, no model is loaded.
        """
        self.model = None
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """
        Load a model.
        
        Parameters:
        -----------
        model_path : str
            Path to the model.
        """
        try:
            # Check if TFLite model
            if model_path.endswith('.tflite'):
                self.model = self._load_tflite_model(model_path)
            else:
                self.model = load_model(model_path)
                
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def _load_tflite_model(self, model_path):
        """
        Load a TFLite model.
        
        Parameters:
        -----------
        model_path : str
            Path to the TFLite model.
            
        Returns:
        --------
        interpreter : tf.lite.Interpreter
            TFLite interpreter.
        """
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            return interpreter
            
        except Exception as e:
            logger.error(f"Error loading TFLite model: {e}")
            raise
            
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
            
        Returns:
        --------
        processed_data : array-like
            Preprocessed ECG data.
        """
        # Basic preprocessing
        # In practice, you would implement more sophisticated preprocessing
        
        # Ensure data is numpy array
        ecg_data = np.array(ecg_data)
        
        # Normalize
        ecg_data = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + 1e-6)
        
        # Reshape if needed
        if len(ecg_data.shape) == 2:
            # Add batch dimension
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        return ecg_data
    
    def predict(self, ecg_data):
        """
        Make a prediction.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
            
        Returns:
        --------
        prediction : array-like
            Prediction.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        try:
            # Preprocess data
            processed_data = self.preprocess_ecg(ecg_data)
            
            # Check if TFLite model
            if isinstance(self.model, tf.lite.Interpreter):
                return self._predict_tflite(processed_data)
            else:
                return self.model.predict(processed_data)
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
            
    def _predict_tflite(self, processed_data):
        """
        Make a prediction using TFLite model.
        
        Parameters:
        -----------
        processed_data : array-like
            Preprocessed ECG data.
            
        Returns:
        --------
        prediction : array-like
            Prediction.
        """
        # Get input and output details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Set input tensor
        self.model.set_tensor(input_details[0]['index'], processed_data.astype(np.float32))
        
        # Run inference
        self.model.invoke()
        
        # Get output tensor
        prediction = self.model.get_tensor(output_details[0]['index'])
        
        return prediction
    
    def start_processing_thread(self):
        """
        Start the processing thread.
        """
        if self.is_processing:
            logger.warning("Processing thread already running.")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Processing thread started.")
        
    def stop_processing_thread(self):
        """
        Stop the processing thread.
        """
        if not self.is_processing:
            logger.warning("Processing thread not running.")
            return
            
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        logger.info("Processing thread stopped.")
        
    def _processing_worker(self):
        """
        Worker function for processing thread.
        """
        while self.is_processing:
            try:
                # Get item from queue with timeout
                try:
                    item = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Process item
                ecg_data = item['ecg_data']
                case_id = item['case_id']
                
                # Make prediction
                prediction = self.predict(ecg_data)
                
                # Put result in results queue
                result = {
                    'case_id': case_id,
                    'prediction': prediction,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.results_queue.put(result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                
    def add_to_queue(self, ecg_data, case_id):
        """
        Add an item to the processing queue.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        case_id : str
            Case ID.
        """
        item = {
            'ecg_data': ecg_data,
            'case_id': case_id
        }
        
        self.processing_queue.put(item)
        logger.info(f"Added case {case_id} to processing queue.")
        
    def get_result(self, block=True, timeout=None):
        """
        Get a result from the results queue.
        
        Parameters:
        -----------
        block : bool
            Whether to block until a result is available. Default is True.
        timeout : float or None
            Timeout in seconds. If None, waits indefinitely. Default is None.
            
        Returns:
        --------
        result : dict or None
            Result dictionary, or None if timeout.
        """
        try:
            return self.results_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def save_results_to_file(self, results, path):
        """
        Save results to a file.
        
        Parameters:
        -----------
        results : list
            List of result dictionaries.
        path : str
            Path to save the results.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save results
            with open(path, 'w') as f:
                json.dump(results, f)
                
            logger.info(f"Results saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    def load_results_from_file(self, path):
        """
        Load results from a file.
        
        Parameters:
        -----------
        path : str
            Path to load the results from.
            
        Returns:
        --------
        results : list
            List of result dictionaries.
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Results file not found: {path}")
                return []
                
            # Load results
            with open(path, 'r') as f:
                results = json.load(f)
                
            logger.info(f"Results loaded from {path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise


class RemoteConsultation:
    """
    Class for remote consultation features.
    """
    
    def __init__(self, server_url=None, api_key=None):
        """
        Initialize the remote consultation.
        
        Parameters:
        -----------
        server_url : str or None
            URL of the remote server. If None, uses offline mode.
        api_key : str or None
            API key for authentication. If None, uses offline mode.
        """
        self.server_url = server_url
        self.api_key = api_key
        self.offline_mode = server_url is None or api_key is None
        self.pending_cases = []
        self.consultation_results = []
        
    def set_server_url(self, server_url):
        """
        Set the server URL.
        
        Parameters:
        -----------
        server_url : str
            URL of the remote server.
        """
        self.server_url = server_url
        self.offline_mode = self.server_url is None or self.api_key is None
        
    def set_api_key(self, api_key):
        """
        Set the API key.
        
        Parameters:
        -----------
        api_key : str
            API key for authentication.
        """
        self.api_key = api_key
        self.offline_mode = self.server_url is None or self.api_key is None
        
    def submit_case(self, case_data, priority='normal'):
        """
        Submit a case for remote consultation.
        
        Parameters:
        -----------
        case_data : dict
            Case data.
        priority : str
            Priority level. Options: 'low', 'normal', 'high', 'urgent'. Default is 'normal'.
            
        Returns:
        --------
        case_id : str or None
            Case ID if successful, None if offline.
        """
        if self.offline_mode:
            # Store case for later submission
            case_id = f"offline_{len(self.pending_cases) + 1}"
            case_data['case_id'] = case_id
            case_data['priority'] = priority
            case_data['submission_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            case_data['status'] = 'pending'
            
            self.pending_cases.append(case_data)
            
            logger.info(f"Case {case_id} stored for later submission (offline mode).")
            return case_id
            
        try:
            # Prepare request
            headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json'
            }
            
            payload = {
                'case_data': case_data,
                'priority': priority
            }
            
            # Send request
            response = requests.post(
                f"{self.server_url}/api/cases",
                headers=headers,
                json=payload
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                case_id = result.get('case_id')
                
                logger.info(f"Case submitted successfully. Case ID: {case_id}")
                return case_id
            else:
                logger.error(f"Error submitting case: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting case: {e}")
            
            # Store case for later submission
            case_id = f"offline_{len(self.pending_cases) + 1}"
            case_data['case_id'] = case_id
            case_data['priority'] = priority
            case_data['submission_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            case_data['status'] = 'pending'
            
            self.pending_cases.append(case_data)
            
            logger.info(f"Case {case_id} stored for later submission (connection error).")
            return case_id
            
    def check_case_status(self, case_id):
        """
        Check the status of a case.
        
        Parameters:
        -----------
        case_id : str
            Case ID.
            
        Returns:
        --------
        status : dict or None
            Status dictionary if successful, None if error or offline.
        """
        if self.offline_mode or case_id.startswith('offline_'):
            # Check pending cases
            for case in self.pending_cases:
                if case['case_id'] == case_id:
                    return {
                        'case_id': case_id,
                        'status': case['status'],
                        'submission_time': case['submission_time']
                    }
                    
            return None
            
        try:
            # Prepare request
            headers = {
                'Authorization': f"Bearer {self.api_key}"
            }
            
            # Send request
            response = requests.get(
                f"{self.server_url}/api/cases/{case_id}/status",
                headers=headers
            )
            
            # Check response
            if response.status_code == 200:
                status = response.json()
                
                logger.info(f"Case status retrieved successfully. Status: {status['status']}")
                return status
            else:
                logger.error(f"Error checking case status: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking case status: {e}")
            return None
            
    def get_consultation_result(self, case_id):
        """
        Get the consultation result for a case.
        
        Parameters:
        -----------
        case_id : str
            Case ID.
            
        Returns:
        --------
        result : dict or None
            Result dictionary if successful, None if error or offline.
        """
        if self.offline_mode or case_id.startswith('offline_'):
            # Check consultation results
            for result in self.consultation_results:
                if result['case_id'] == case_id:
                    return result
                    
            return None
            
        try:
            # Prepare request
            headers = {
                'Authorization': f"Bearer {self.api_key}"
            }
            
            # Send request
            response = requests.get(
                f"{self.server_url}/api/cases/{case_id}/result",
                headers=headers
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                
                logger.info(f"Consultation result retrieved successfully.")
                return result
            else:
                logger.error(f"Error getting consultation result: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting consultation result: {e}")
            return None
            
    def sync_pending_cases(self):
        """
        Synchronize pending cases with the server.
        
        Returns:
        --------
        n_synced : int
            Number of cases successfully synchronized.
        """
        if self.offline_mode:
            logger.warning("Cannot sync cases in offline mode.")
            return 0
            
        if not self.pending_cases:
            logger.info("No pending cases to sync.")
            return 0
            
        n_synced = 0
        
        for i, case in enumerate(self.pending_cases[:]):
            try:
                # Submit case
                case_id = self.submit_case(case, case['priority'])
                
                if case_id and not case_id.startswith('offline_'):
                    # Case submitted successfully
                    n_synced += 1
                    
                    # Remove from pending cases
                    self.pending_cases.remove(case)
                    
            except Exception as e:
                logger.error(f"Error syncing case: {e}")
                
        logger.info(f"Synced {n_synced} pending cases.")
        return n_synced
    
    def save_pending_cases(self, path):
        """
        Save pending cases to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the pending cases.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save pending cases
            with open(path, 'w') as f:
                json.dump(self.pending_cases, f)
                
            logger.info(f"Pending cases saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving pending cases: {e}")
            raise
            
    def load_pending_cases(self, path):
        """
        Load pending cases from a file.
        
        Parameters:
        -----------
        path : str
            Path to load the pending cases from.
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Pending cases file not found: {path}")
                return
                
            # Load pending cases
            with open(path, 'r') as f:
                self.pending_cases = json.load(f)
                
            logger.info(f"Loaded {len(self.pending_cases)} pending cases from {path}")
            
        except Exception as e:
            logger.error(f"Error loading pending cases: {e}")
            raise
            
    def save_consultation_results(self, path):
        """
        Save consultation results to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the consultation results.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save consultation results
            with open(path, 'w') as f:
                json.dump(self.consultation_results, f)
                
            logger.info(f"Consultation results saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving consultation results: {e}")
            raise
            
    def load_consultation_results(self, path):
        """
        Load consultation results from a file.
        
        Parameters:
        -----------
        path : str
            Path to load the consultation results from.
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Consultation results file not found: {path}")
                return
                
            # Load consultation results
            with open(path, 'r') as f:
                self.consultation_results = json.load(f)
                
            logger.info(f"Loaded {len(self.consultation_results)} consultation results from {path}")
            
        except Exception as e:
            logger.error(f"Error loading consultation results: {e}")
            raise


class RemoteHealthcareSystem:
    """
    Class for managing the remote healthcare system.
    """
    
    def __init__(self, model_path=None, server_url=None, api_key=None):
        """
        Initialize the remote healthcare system.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the model. If None, no model is loaded.
        server_url : str or None
            URL of the remote server. If None, uses offline mode.
        api_key : str or None
            API key for authentication. If None, uses offline mode.
        """
        # Initialize components
        self.model_optimizer = ModelOptimizer(model_path)
        self.offline_processor = OfflineProcessor(model_path)
        self.remote_consultation = RemoteConsultation(server_url, api_key)
        
        # Start processing thread
        self.offline_processor.start_processing_thread()
        
    def optimize_model(self, quantization_type='dynamic', target_sparsity=0.5):
        """
        Optimize the model.
        
        Parameters:
        -----------
        quantization_type : str
            Type of quantization. Options: 'dynamic', 'float16', 'int8'. Default is 'dynamic'.
        target_sparsity : float
            Target sparsity for pruning. Default is 0.5.
            
        Returns:
        --------
        optimized_model : Model
            Optimized model.
        """
        # Quantize model
        quantized_model = self.model_optimizer.quantize_model(quantization_type)
        
        # Prune model
        pruned_model = self.model_optimizer.prune_model(target_sparsity)
        
        # Return optimized model
        return self.model_optimizer.optimized_model
    
    def process_ecg(self, ecg_data, case_id=None, metadata=None):
        """
        Process ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        case_id : str or None
            Case ID. If None, generates a new ID.
        metadata : dict or None
            Additional metadata. If None, uses empty dict.
            
        Returns:
        --------
        result : dict
            Processing result.
        """
        # Generate case ID if not provided
        if case_id is None:
            case_id = f"case_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        # Set metadata
        if metadata is None:
            metadata = {}
            
        # Add to processing queue
        self.offline_processor.add_to_queue(ecg_data, case_id)
        
        # Wait for result
        result = self.offline_processor.get_result(timeout=10)
        
        if result is None:
            # No result yet, return pending status
            return {
                'case_id': case_id,
                'status': 'pending',
                'message': 'ECG data added to processing queue.'
            }
            
        # Add metadata to result
        result['metadata'] = metadata
        
        # Check if case should be submitted for remote consultation
        if metadata.get('requires_consultation', False):
            # Submit case
            consultation_case_id = self.remote_consultation.submit_case(
                {
                    'ecg_data': ecg_data.tolist() if isinstance(ecg_data, np.ndarray) else ecg_data,
                    'prediction': result['prediction'].tolist() if isinstance(result['prediction'], np.ndarray) else result['prediction'],
                    'metadata': metadata
                },
                priority=metadata.get('priority', 'normal')
            )
            
            result['consultation_case_id'] = consultation_case_id
            
        return result
    
    def check_consultation_status(self, case_id):
        """
        Check the status of a consultation case.
        
        Parameters:
        -----------
        case_id : str
            Case ID.
            
        Returns:
        --------
        status : dict or None
            Status dictionary if successful, None if error.
        """
        return self.remote_consultation.check_case_status(case_id)
    
    def get_consultation_result(self, case_id):
        """
        Get the consultation result for a case.
        
        Parameters:
        -----------
        case_id : str
            Case ID.
            
        Returns:
        --------
        result : dict or None
            Result dictionary if successful, None if error.
        """
        return self.remote_consultation.get_consultation_result(case_id)
    
    def sync_with_server(self):
        """
        Synchronize with the server.
        
        Returns:
        --------
        n_synced : int
            Number of cases successfully synchronized.
        """
        return self.remote_consultation.sync_pending_cases()
    
    def save_state(self, directory):
        """
        Save the system state.
        
        Parameters:
        -----------
        directory : str
            Directory to save the state.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save optimized model
            if self.model_optimizer.optimized_model is not None:
                self.model_optimizer.save_optimized_model(os.path.join(directory, 'optimized_model.h5'))
                self.model_optimizer.export_tflite_model(os.path.join(directory, 'optimized_model.tflite'))
                
            # Save pending cases
            self.remote_consultation.save_pending_cases(os.path.join(directory, 'pending_cases.json'))
            
            # Save consultation results
            self.remote_consultation.save_consultation_results(os.path.join(directory, 'consultation_results.json'))
            
            logger.info(f"System state saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            raise
            
    def load_state(self, directory):
        """
        Load the system state.
        
        Parameters:
        -----------
        directory : str
            Directory to load the state from.
        """
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                logger.warning(f"State directory not found: {directory}")
                return
                
            # Load optimized model
            model_path = os.path.join(directory, 'optimized_model.h5')
            tflite_path = os.path.join(directory, 'optimized_model.tflite')
            
            if os.path.exists(model_path):
                self.model_optimizer.load_model(model_path)
                self.offline_processor.load_model(model_path)
            elif os.path.exists(tflite_path):
                self.offline_processor.load_model(tflite_path)
                
            # Load pending cases
            self.remote_consultation.load_pending_cases(os.path.join(directory, 'pending_cases.json'))
            
            # Load consultation results
            self.remote_consultation.load_consultation_results(os.path.join(directory, 'consultation_results.json'))
            
            logger.info(f"System state loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            raise
            
    def shutdown(self):
        """
        Shutdown the system.
        """
        try:
            # Stop processing thread
            self.offline_processor.stop_processing_thread()
            
            logger.info("System shutdown complete.")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Create a simple model for demonstration
    def create_demo_model():
        inputs = tf.keras.layers.Input(shape=(100, 12))
        x = tf.keras.layers.Conv1D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    # Create and save demo model
    model = create_demo_model()
    model_path = 'demo_model.h5'
    model.save(model_path)
    
    print(f"Demo model created and saved to {model_path}")
    
    # Create remote healthcare system
    system = RemoteHealthcareSystem(model_path)
    
    # Optimize model
    optimized_model = system.optimize_model()
    
    print("Model optimized")
    
    # Generate synthetic ECG data
    ecg_data = np.random.randn(100, 12)
    
    # Process ECG data
    result = system.process_ecg(ecg_data, metadata={'patient_id': '12345'})
    
    print(f"ECG processed: {result}")
    
    # Save system state
    system.save_state('remote_healthcare_state')
    
    print("System state saved")
    
    # Shutdown system
    system.shutdown()
    
    print("System shutdown complete")

