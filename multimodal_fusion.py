"""
CardioInsight AI - Multimodal Fusion Module

This module provides tools for fusing ECG data with other clinical data (e.g., patient history, symptoms).
It includes various fusion strategies and models for multimodal learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json

class MultimodalFusion:
    """
    Base class for multimodal fusion models.
    """
    
    def __init__(self, fusion_strategy='concat', model_name='multimodal_fusion'):
        """
        Initialize the multimodal fusion model.
        
        Parameters:
        -----------
        fusion_strategy : str
            Fusion strategy. Options: 'concat', 'attention', 'weighted'. Default is 'concat'.
        model_name : str
            Name of the model. Default is 'multimodal_fusion'.
        """
        self.fusion_strategy = fusion_strategy
        self.model_name = model_name
        self.model = None
        self.history = None
        self.classes_ = None
        self.ecg_scaler = StandardScaler()
        self.tabular_scaler = StandardScaler()
        
    def build_model(self, ecg_shape, tabular_shape, num_classes):
        """
        Build the multimodal fusion model.
        
        Parameters:
        -----------
        ecg_shape : tuple
            Shape of ECG data (samples, leads).
        tabular_shape : int
            Number of tabular features.
        num_classes : int
            Number of classes.
            
        Returns:
        --------
        model : Model
            Keras model.
        """
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def compile_model(self, learning_rate=0.001, metrics=None):
        """
        Compile the model.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for the optimizer. Default is 0.001.
        metrics : list or None
            List of metrics to monitor. If None, uses ['accuracy'].
            
        Returns:
        --------
        self : object
            Compiled model.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if metrics is None:
            metrics = ['accuracy']
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        return self
    
    def fit(self, X_ecg, X_tabular, y, validation_data=None, batch_size=32, epochs=50, callbacks=None, verbose=1):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X_ecg : array-like
            ECG data.
        X_tabular : array-like
            Tabular data.
        y : array-like
            Target labels.
        validation_data : tuple or None
            Validation data ((X_ecg_val, X_tabular_val), y_val). If None, uses 20% of training data.
        batch_size : int
            Batch size. Default is 32.
        epochs : int
            Number of epochs. Default is 50.
        callbacks : list or None
            List of callbacks. If None, uses default callbacks.
        verbose : int
            Verbosity mode. Default is 1.
            
        Returns:
        --------
        self : object
            Fitted model.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Scale tabular data
        X_tabular_scaled = self.tabular_scaler.fit_transform(X_tabular)
        
        # Convert labels to one-hot encoding
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            self.classes_ = np.unique(y)
            num_classes = len(self.classes_)
            
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # Already one-hot encoded
            y_encoded = y
            num_classes = y.shape[1]
            self.classes_ = np.arange(num_classes)
            
        # Create validation split if not provided
        if validation_data is None:
            # Split data
            X_ecg_train, X_ecg_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
                X_ecg, X_tabular_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        else:
            X_ecg_train, X_tab_train, y_train = X_ecg, X_tabular_scaled, y_encoded
            (X_ecg_val, X_tab_val), y_val = validation_data
            
            # Scale validation tabular data
            X_tab_val = self.tabular_scaler.transform(X_tab_val)
            
            # Convert validation labels to one-hot if needed
            if len(y_val.shape) == 1 or y_val.shape[1] == 1:
                y_val = encoder.transform(y_val.reshape(-1, 1))
                
        # Create default callbacks if not provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                ),
                ModelCheckpoint(
                    f"{self.model_name}_best.h5", monitor='val_loss',
                    save_best_only=True, save_weights_only=False
                )
            ]
            
        # Fit model
        self.history = self.model.fit(
            [X_ecg_train, X_tab_train], y_train,
            validation_data=([X_ecg_val, X_tab_val], y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X_ecg, X_tabular):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X_ecg : array-like
            ECG data.
        X_tabular : array-like
            Tabular data.
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale tabular data
        X_tabular_scaled = self.tabular_scaler.transform(X_tabular)
        
        # Get predicted probabilities
        y_proba = self.model.predict([X_ecg, X_tabular_scaled])
        
        # Convert to class labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Map to original classes if available
        if self.classes_ is not None:
            y_pred = self.classes_[y_pred]
            
        return y_pred
    
    def predict_proba(self, X_ecg, X_tabular):
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X_ecg : array-like
            ECG data.
        X_tabular : array-like
            Tabular data.
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale tabular data
        X_tabular_scaled = self.tabular_scaler.transform(X_tabular)
        
        return self.model.predict([X_ecg, X_tabular_scaled])
    
    def evaluate(self, X_ecg, X_tabular, y, metrics=None):
        """
        Evaluate the model on the given data.
        
        Parameters:
        -----------
        X_ecg : array-like
            ECG data.
        X_tabular : array-like
            Tabular data.
        y : array-like
            True labels.
        metrics : list or None
            List of metrics to compute. If None, computes accuracy, precision, recall, and F1.
            
        Returns:
        --------
        results : dict
            Dictionary of evaluation results.
        """
        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
        # Scale tabular data
        X_tabular_scaled = self.tabular_scaler.transform(X_tabular)
        
        # Convert labels to one-hot if needed
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
            y_true = y
        else:
            # Already one-hot encoded
            y_encoded = y
            y_true = np.argmax(y, axis=1)
            
        # Get predictions
        y_pred = self.predict(X_ecg, X_tabular)
        
        # Compute metrics
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
            
        if 'precision' in metrics:
            results['precision'] = precision_score(y_true, y_pred, average='weighted')
            
        if 'recall' in metrics:
            results['recall'] = recall_score(y_true, y_pred, average='weighted')
            
        if 'f1' in metrics:
            results['f1'] = f1_score(y_true, y_pred, average='weighted')
            
        # Add model evaluation metrics
        model_metrics = self.model.evaluate([X_ecg, X_tabular_scaled], y_encoded, verbose=0)
        for i, metric_name in enumerate(self.model.metrics_names):
            results[metric_name] = model_metrics[i]
            
        return results
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save scalers
        scalers_filepath = os.path.splitext(filepath)[0] + '_scalers.npz'
        np.savez(
            scalers_filepath,
            tabular_scaler_mean=self.tabular_scaler.mean_,
            tabular_scaler_scale=self.tabular_scaler.scale_
        )
        
        # Save additional information
        info_filepath = os.path.splitext(filepath)[0] + '_info.json'
        model_info = {
            'fusion_strategy': self.fusion_strategy,
            'model_name': self.model_name,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }
        
        with open(info_filepath, 'w') as f:
            json.dump(model_info, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        model : MultimodalFusion
            Loaded model.
        """
        # Load model info
        info_filepath = os.path.splitext(filepath)[0] + '_info.json'
        with open(info_filepath, 'r') as f:
            model_info = json.load(f)
            
        # Create instance
        instance = cls(
            fusion_strategy=model_info['fusion_strategy'],
            model_name=model_info['model_name']
        )
        
        # Load model
        instance.model = tf.keras.models.load_model(filepath)
        
        # Load scalers
        scalers_filepath = os.path.splitext(filepath)[0] + '_scalers.npz'
        scalers_data = np.load(scalers_filepath)
        
        instance.tabular_scaler = StandardScaler()
        instance.tabular_scaler.mean_ = scalers_data['tabular_scaler_mean']
        instance.tabular_scaler.scale_ = scalers_data['tabular_scaler_scale']
        
        # Set classes
        instance.classes_ = np.array(model_info['classes']) if model_info['classes'] is not None else None
        
        return instance
    
    def plot_training_history(self):
        """
        Plot training history.
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        if self.history is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='train')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='validation')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='train')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='validation')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        fig.tight_layout()
        return fig


class EarlyFusion(MultimodalFusion):
    """
    Early fusion model for multimodal data.
    """
    
    def __init__(self, fusion_strategy='concat', model_name='early_fusion'):
        """
        Initialize the early fusion model.
        
        Parameters:
        -----------
        fusion_strategy : str
            Fusion strategy. Options: 'concat', 'attention', 'weighted'. Default is 'concat'.
        model_name : str
            Name of the model. Default is 'early_fusion'.
        """
        super().__init__(fusion_strategy, model_name)
    
    def build_model(self, ecg_shape, tabular_shape, num_classes):
        """
        Build the early fusion model.
        
        Parameters:
        -----------
        ecg_shape : tuple
            Shape of ECG data (samples, leads).
        tabular_shape : int
            Number of tabular features.
        num_classes : int
            Number of classes.
            
        Returns:
        --------
        model : Model
            Keras model.
        """
        # ECG input
        ecg_input = Input(shape=ecg_shape, name='ecg_input')
        
        # Tabular input
        tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
        
        # Process ECG data
        x_ecg = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(ecg_input)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        x_ecg = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        x_ecg = tf.keras.layers.GlobalAveragePooling1D()(x_ecg)
        
        # Process tabular data
        x_tab = Dense(64)(tabular_input)
        x_tab = BatchNormalization()(x_tab)
        x_tab = Activation('relu')(x_tab)
        
        # Fusion
        if self.fusion_strategy == 'concat':
            # Concatenation fusion
            x = Concatenate()([x_ecg, x_tab])
        elif self.fusion_strategy == 'attention':
            # Attention fusion
            attention_weights = Dense(1, activation='sigmoid')(x_tab)
            x_ecg_weighted = Multiply()([x_ecg, attention_weights])
            x = Concatenate()([x_ecg_weighted, x_tab])
        elif self.fusion_strategy == 'weighted':
            # Weighted fusion
            x_ecg_proj = Dense(64)(x_ecg)
            x_tab_proj = Dense(64)(x_tab)
            x = Add()([x_ecg_proj, x_tab_proj])
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
            
        # Dense layers
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[ecg_input, tabular_input], outputs=outputs, name=self.model_name)
        
        self.model = model
        return model


class LateFusion(MultimodalFusion):
    """
    Late fusion model for multimodal data.
    """
    
    def __init__(self, fusion_strategy='concat', model_name='late_fusion'):
        """
        Initialize the late fusion model.
        
        Parameters:
        -----------
        fusion_strategy : str
            Fusion strategy. Options: 'concat', 'average', 'weighted'. Default is 'concat'.
        model_name : str
            Name of the model. Default is 'late_fusion'.
        """
        super().__init__(fusion_strategy, model_name)
    
    def build_model(self, ecg_shape, tabular_shape, num_classes):
        """
        Build the late fusion model.
        
        Parameters:
        -----------
        ecg_shape : tuple
            Shape of ECG data (samples, leads).
        tabular_shape : int
            Number of tabular features.
        num_classes : int
            Number of classes.
            
        Returns:
        --------
        model : Model
            Keras model.
        """
        # ECG input
        ecg_input = Input(shape=ecg_shape, name='ecg_input')
        
        # Tabular input
        tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
        
        # ECG branch
        x_ecg = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(ecg_input)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        x_ecg = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        x_ecg = tf.keras.layers.GlobalAveragePooling1D()(x_ecg)
        
        x_ecg = Dense(128)(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = Dropout(0.5)(x_ecg)
        
        # ECG output
        ecg_outputs = Dense(num_classes, activation='softmax', name='ecg_output')(x_ecg)
        
        # Tabular branch
        x_tab = Dense(64)(tabular_input)
        x_tab = BatchNormalization()(x_tab)
        x_tab = Activation('relu')(x_tab)
        
        x_tab = Dense(128)(x_tab)
        x_tab = BatchNormalization()(x_tab)
        x_tab = Activation('relu')(x_tab)
        x_tab = Dropout(0.5)(x_tab)
        
        # Tabular output
        tab_outputs = Dense(num_classes, activation='softmax', name='tab_output')(x_tab)
        
        # Fusion
        if self.fusion_strategy == 'concat':
            # Concatenate predictions and add a final layer
            x = Concatenate()([ecg_outputs, tab_outputs])
            outputs = Dense(num_classes, activation='softmax', name='final_output')(x)
        elif self.fusion_strategy == 'average':
            # Simple average
            outputs = tf.keras.layers.Average(name='final_output')([ecg_outputs, tab_outputs])
        elif self.fusion_strategy == 'weighted':
            # Weighted average
            weight_ecg = Dense(1, activation='sigmoid', name='weight_ecg')(x_ecg)
            weight_tab = Dense(1, activation='sigmoid', name='weight_tab')(x_tab)
            
            # Normalize weights
            weights = Concatenate()([weight_ecg, weight_tab])
            weights = tf.keras.layers.Softmax(axis=1)(weights)
            
            # Apply weights
            weighted_ecg = Multiply()([ecg_outputs, weights[:, 0:1]])
            weighted_tab = Multiply()([tab_outputs, weights[:, 1:2]])
            
            # Sum
            outputs = Add(name='final_output')([weighted_ecg, weighted_tab])
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
            
        # Create model
        model = Model(inputs=[ecg_input, tabular_input], outputs=outputs, name=self.model_name)
        
        self.model = model
        return model


class HybridFusion(MultimodalFusion):
    """
    Hybrid fusion model for multimodal data.
    """
    
    def __init__(self, fusion_strategy='concat', model_name='hybrid_fusion'):
        """
        Initialize the hybrid fusion model.
        
        Parameters:
        -----------
        fusion_strategy : str
            Fusion strategy. Options: 'concat', 'attention', 'cross_attention'. Default is 'concat'.
        model_name : str
            Name of the model. Default is 'hybrid_fusion'.
        """
        super().__init__(fusion_strategy, model_name)
    
    def build_model(self, ecg_shape, tabular_shape, num_classes):
        """
        Build the hybrid fusion model.
        
        Parameters:
        -----------
        ecg_shape : tuple
            Shape of ECG data (samples, leads).
        tabular_shape : int
            Number of tabular features.
        num_classes : int
            Number of classes.
            
        Returns:
        --------
        model : Model
            Keras model.
        """
        # ECG input
        ecg_input = Input(shape=ecg_shape, name='ecg_input')
        
        # Tabular input
        tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
        
        # ECG branch - feature extraction
        x_ecg = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(ecg_input)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        x_ecg = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg_features = tf.keras.layers.MaxPooling1D(pool_size=2)(x_ecg)
        
        # Tabular branch - feature extraction
        x_tab = Dense(64)(tabular_input)
        x_tab = BatchNormalization()(x_tab)
        x_tab = Activation('relu')(x_tab)
        x_tab_features = Dropout(0.2)(x_tab)
        
        # Mid-level fusion
        if self.fusion_strategy == 'concat':
            # Global pooling for ECG
            x_ecg_pooled = tf.keras.layers.GlobalAveragePooling1D()(x_ecg_features)
            
            # Concatenate
            x = Concatenate()([x_ecg_pooled, x_tab_features])
        elif self.fusion_strategy == 'attention':
            # Create attention weights from tabular data
            attention_dense = Dense(64)(x_tab_features)
            attention_dense = Activation('relu')(attention_dense)
            attention_weights = Dense(x_ecg_features.shape[1], activation='softmax')(attention_dense)
            
            # Reshape for broadcasting
            attention_weights = tf.keras.layers.Reshape((x_ecg_features.shape[1], 1))(attention_weights)
            
            # Apply attention to ECG features
            x_ecg_weighted = Multiply()([x_ecg_features, attention_weights])
            
            # Global pooling
            x_ecg_pooled = tf.keras.layers.GlobalAveragePooling1D()(x_ecg_weighted)
            
            # Concatenate
            x = Concatenate()([x_ecg_pooled, x_tab_features])
        elif self.fusion_strategy == 'cross_attention':
            # Global pooling for ECG
            x_ecg_pooled = tf.keras.layers.GlobalAveragePooling1D()(x_ecg_features)
            
            # Cross-attention: ECG -> Tabular
            ecg_to_tab_attn = Dense(64)(x_ecg_pooled)
            ecg_to_tab_attn = Activation('sigmoid')(ecg_to_tab_attn)
            x_tab_weighted = Multiply()([x_tab_features, ecg_to_tab_attn])
            
            # Cross-attention: Tabular -> ECG
            tab_to_ecg_attn = Dense(x_ecg_features.shape[1])(x_tab_features)
            tab_to_ecg_attn = Activation('sigmoid')(tab_to_ecg_attn)
            tab_to_ecg_attn = tf.keras.layers.Reshape((x_ecg_features.shape[1], 1))(tab_to_ecg_attn)
            x_ecg_weighted = Multiply()([x_ecg_features, tab_to_ecg_attn])
            x_ecg_pooled = tf.keras.layers.GlobalAveragePooling1D()(x_ecg_weighted)
            
            # Concatenate
            x = Concatenate()([x_ecg_pooled, x_tab_weighted])
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
            
        # Dense layers
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[ecg_input, tabular_input], outputs=outputs, name=self.model_name)
        
        self.model = model
        return model


class TabularFeatureExtractor:
    """
    Class for extracting and processing tabular features from patient data.
    """
    
    def __init__(self):
        """
        Initialize the tabular feature extractor.
        """
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.feature_names = None
        
    def fit_transform(self, data, categorical_cols=None, numerical_cols=None):
        """
        Fit the feature extractor and transform the data.
        
        Parameters:
        -----------
        data : DataFrame
            Patient data.
        categorical_cols : list or None
            List of categorical column names. If None, infers from data.
        numerical_cols : list or None
            List of numerical column names. If None, infers from data.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        # Infer column types if not provided
        if categorical_cols is None and numerical_cols is None:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            
        # Process categorical features
        categorical_features = []
        for col in categorical_cols:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(data[col].values.reshape(-1, 1))
            categorical_features.append(encoded)
            self.categorical_encoders[col] = encoder
            
        # Process numerical features
        if numerical_cols:
            numerical_features = self.numerical_scaler.fit_transform(data[numerical_cols])
        else:
            numerical_features = np.empty((len(data), 0))
            
        # Combine features
        if categorical_features:
            features = np.hstack([numerical_features] + categorical_features)
        else:
            features = numerical_features
            
        # Store feature names
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(numerical_cols)
        
        # Add categorical feature names
        for col in categorical_cols:
            encoder = self.categorical_encoders[col]
            categories = encoder.categories_[0]
            for category in categories:
                feature_names.append(f"{col}_{category}")
                
        self.feature_names = feature_names
        
        return features
    
    def transform(self, data):
        """
        Transform the data using the fitted feature extractor.
        
        Parameters:
        -----------
        data : DataFrame
            Patient data.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        if not self.categorical_encoders:
            raise ValueError("Feature extractor not fitted. Call fit_transform() first.")
            
        # Process categorical features
        categorical_features = []
        for col, encoder in self.categorical_encoders.items():
            if col in data.columns:
                encoded = encoder.transform(data[col].values.reshape(-1, 1))
                categorical_features.append(encoded)
            else:
                # Create empty array with correct shape
                n_categories = len(encoder.categories_[0])
                empty_encoded = np.zeros((len(data), n_categories))
                categorical_features.append(empty_encoded)
                
        # Process numerical features
        numerical_cols = [col for col in data.columns if col not in self.categorical_encoders]
        if numerical_cols:
            numerical_features = self.numerical_scaler.transform(data[numerical_cols])
        else:
            numerical_features = np.empty((len(data), 0))
            
        # Combine features
        if categorical_features:
            features = np.hstack([numerical_features] + categorical_features)
        else:
            features = numerical_features
            
        return features
    
    def get_feature_names(self):
        """
        Get the names of the extracted features.
        
        Returns:
        --------
        feature_names : list
            List of feature names.
        """
        if self.feature_names is None:
            raise ValueError("Feature extractor not fitted. Call fit_transform() first.")
            
        return self.feature_names


class TextFeatureExtractor:
    """
    Class for extracting features from text data (e.g., symptoms, medical notes).
    """
    
    def __init__(self, method='tfidf'):
        """
        Initialize the text feature extractor.
        
        Parameters:
        -----------
        method : str
            Feature extraction method. Options: 'tfidf', 'count', 'embedding'. Default is 'tfidf'.
        """
        self.method = method
        self.vectorizer = None
        self.embedding_model = None
        
    def fit_transform(self, texts):
        """
        Fit the feature extractor and transform the texts.
        
        Parameters:
        -----------
        texts : list
            List of text strings.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        if self.method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            features = self.vectorizer.fit_transform(texts).toarray()
        elif self.method == 'count':
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(max_features=100, stop_words='english')
            features = self.vectorizer.fit_transform(texts).toarray()
        elif self.method == 'embedding':
            try:
                import tensorflow_hub as hub
                self.embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                features = self.embedding_model(texts).numpy()
            except ImportError:
                raise ImportError("tensorflow_hub is required for embedding method.")
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return features
    
    def transform(self, texts):
        """
        Transform the texts using the fitted feature extractor.
        
        Parameters:
        -----------
        texts : list
            List of text strings.
            
        Returns:
        --------
        features : array-like
            Extracted features.
        """
        if self.method in ['tfidf', 'count']:
            if self.vectorizer is None:
                raise ValueError("Feature extractor not fitted. Call fit_transform() first.")
            features = self.vectorizer.transform(texts).toarray()
        elif self.method == 'embedding':
            if self.embedding_model is None:
                raise ValueError("Feature extractor not fitted. Call fit_transform() first.")
            features = self.embedding_model(texts).numpy()
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return features
    
    def get_feature_names(self):
        """
        Get the names of the extracted features.
        
        Returns:
        --------
        feature_names : list
            List of feature names.
        """
        if self.method in ['tfidf', 'count']:
            if self.vectorizer is None:
                raise ValueError("Feature extractor not fitted. Call fit_transform() first.")
            return self.vectorizer.get_feature_names_out()
        elif self.method == 'embedding':
            # Embeddings don't have interpretable feature names
            return [f"embedding_{i}" for i in range(512)]  # Universal Sentence Encoder has 512 dimensions
        else:
            raise ValueError(f"Unsupported method: {self.method}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 100
    n_timesteps = 1000
    n_leads = 12
    n_tabular_features = 20
    n_classes = 5
    
    # ECG data
    X_ecg = np.random.randn(n_samples, n_timesteps, n_leads)
    
    # Tabular data
    X_tabular = np.random.randn(n_samples, n_tabular_features)
    
    # Labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Create and train early fusion model
    print("Training early fusion model...")
    early_fusion = EarlyFusion(fusion_strategy='concat')
    early_fusion.build_model(ecg_shape=(n_timesteps, n_leads), tabular_shape=n_tabular_features, num_classes=n_classes)
    early_fusion.compile_model()
    early_fusion.fit(X_ecg, X_tabular, y, epochs=5, batch_size=16, verbose=1)
    
    # Evaluate
    results = early_fusion.evaluate(X_ecg, X_tabular, y)
    print("\nEarly fusion evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
            
    # Create and train hybrid fusion model
    print("\nTraining hybrid fusion model...")
    hybrid_fusion = HybridFusion(fusion_strategy='attention')
    hybrid_fusion.build_model(ecg_shape=(n_timesteps, n_leads), tabular_shape=n_tabular_features, num_classes=n_classes)
    hybrid_fusion.compile_model()
    hybrid_fusion.fit(X_ecg, X_tabular, y, epochs=5, batch_size=16, verbose=1)
    
    # Evaluate
    results = hybrid_fusion.evaluate(X_ecg, X_tabular, y)
    print("\nHybrid fusion evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
            
    # Example with tabular feature extractor
    print("\nTesting tabular feature extractor...")
    # Create synthetic DataFrame
    import pandas as pd
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'condition': np.random.choice(['normal', 'abnormal', 'critical'], n_samples)
    })
    
    # Extract features
    extractor = TabularFeatureExtractor()
    features = extractor.fit_transform(data)
    
    print(f"Extracted {features.shape[1]} features from tabular data")
    print(f"Feature names: {extractor.get_feature_names()}")

