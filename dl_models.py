"""
CardioInsight AI - Deep Learning Models Module

This module provides deep learning models for ECG analysis.
It includes CNN, RNN, and attention-based models for multi-lead ECG classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Activation, Add
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Attention
from tensorflow.keras.layers import Concatenate, Reshape, Permute, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

class ECGDeepLearningModel:
    """
    Base class for deep learning models for ECG analysis.
    """
    
    def __init__(self, input_shape=(5000, 12), num_classes=5, model_name='ecg_model'):
        """
        Initialize the ECG deep learning model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (samples, leads). Default is (5000, 12).
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_model'.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.classes_ = None
        
    def build_model(self):
        """
        Build the model architecture.
        This method should be implemented by subclasses.
        
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
            self.model = self.build_model()
            
        if metrics is None:
            metrics = ['accuracy']
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        return self
    
    def fit(self, X, y, validation_data=None, batch_size=32, epochs=50, callbacks=None, verbose=1):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : array-like
            Training data.
        y : array-like
            Target labels.
        validation_data : tuple or None
            Validation data (X_val, y_val). If None, uses 20% of training data.
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
            self.compile_model()
            
        # Convert labels to one-hot encoding
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            self.classes_ = np.unique(y)
            self.num_classes = len(self.classes_)
            
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # Already one-hot encoded
            y_encoded = y
            self.num_classes = y.shape[1]
            self.classes_ = np.arange(self.num_classes)
            
        # Create validation split if not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = validation_data
            
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
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like
            Features.
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Get predicted probabilities
        y_proba = self.model.predict(X)
        
        # Convert to class labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Map to original classes if available
        if self.classes_ is not None:
            y_pred = self.classes_[y_pred]
            
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like
            Features.
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        return self.model.predict(X)
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the model on the given data.
        
        Parameters:
        -----------
        X : array-like
            Features.
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
            
        # Convert labels to one-hot if needed
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # Already one-hot encoded
            y_encoded = y
            
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_encoded, axis=1)
        
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
            
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
        if 'classification_report' in metrics:
            results['classification_report'] = classification_report(y_true, y_pred)
            
        # Add model evaluation metrics
        model_metrics = self.model.evaluate(X, y_encoded, verbose=0)
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
        
        # Save additional information
        info_filepath = os.path.splitext(filepath)[0] + '_info.json'
        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
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
        model : ECGDeepLearningModel
            Loaded model.
        """
        # Load model
        keras_model = load_model(filepath)
        
        # Load additional information
        info_filepath = os.path.splitext(filepath)[0] + '_info.json'
        with open(info_filepath, 'r') as f:
            model_info = json.load(f)
            
        # Create instance
        instance = cls(
            input_shape=tuple(model_info['input_shape']),
            num_classes=model_info['num_classes'],
            model_name=model_info['model_name']
        )
        
        # Set attributes
        instance.model = keras_model
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
    
    def plot_confusion_matrix(self, X, y, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            True labels.
        normalize : bool
            Whether to normalize the confusion matrix. Default is False.
        title : str or None
            Plot title. If None, uses a default title.
        cmap : colormap
            Colormap to use. Default is plt.cm.Blues.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Convert labels to one-hot if needed
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            # Convert to indices
            y_true = y
        else:
            # Already one-hot encoded
            y_true = np.argmax(y, axis=1)
            
        # Get predictions
        y_pred = self.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = self.classes_ if self.classes_ is not None else np.arange(self.num_classes)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        # Set title
        if title is None:
            title = f'Confusion Matrix ({self.model_name})'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig
    
    def get_attention_maps(self, X):
        """
        Get attention maps for visualization.
        This method should be implemented by subclasses that use attention mechanisms.
        
        Parameters:
        -----------
        X : array-like
            Input data.
            
        Returns:
        --------
        attention_maps : array-like
            Attention maps.
        """
        raise NotImplementedError("This model does not support attention visualization")


class ECGConvNet(ECGDeepLearningModel):
    """
    Convolutional Neural Network for ECG classification.
    """
    
    def __init__(self, input_shape=(5000, 12), num_classes=5, model_name='ecg_convnet'):
        """
        Initialize the ECG ConvNet model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (samples, leads). Default is (5000, 12).
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_convnet'.
        """
        super().__init__(input_shape, num_classes, model_name)
    
    def build_model(self):
        """
        Build the ConvNet model architecture.
        
        Returns:
        --------
        model : Model
            Keras model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First convolutional block
        x = Conv1D(filters=64, kernel_size=5, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Second convolutional block
        x = Conv1D(filters=128, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Third convolutional block
        x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Fourth convolutional block
        x = Conv1D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        return model


class ECGResNet(ECGDeepLearningModel):
    """
    ResNet model for ECG classification.
    """
    
    def __init__(self, input_shape=(5000, 12), num_classes=5, model_name='ecg_resnet'):
        """
        Initialize the ECG ResNet model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (samples, leads). Default is (5000, 12).
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_resnet'.
        """
        super().__init__(input_shape, num_classes, model_name)
    
    def _residual_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True):
        """
        Create a residual block.
        
        Parameters:
        -----------
        x : Tensor
            Input tensor.
        filters : int
            Number of filters.
        kernel_size : int
            Kernel size. Default is 3.
        stride : int
            Stride. Default is 1.
        conv_shortcut : bool
            Whether to use convolution for shortcut. Default is True.
            
        Returns:
        --------
        x : Tensor
            Output tensor.
        """
        shortcut = x
        
        if conv_shortcut:
            shortcut = Conv1D(filters, 1, strides=stride)(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
        # First convolution
        x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second convolution
        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Add shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        return x
    
    def build_model(self):
        """
        Build the ResNet model architecture.
        
        Returns:
        --------
        model : Model
            Keras model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Initial convolution
        x = Conv1D(64, 7, strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64, conv_shortcut=False)
        
        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128, conv_shortcut=False)
        
        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256, conv_shortcut=False)
        
        x = self._residual_block(x, 512, stride=2)
        x = self._residual_block(x, 512, conv_shortcut=False)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layer
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        return model


class ECGAttentionNet(ECGDeepLearningModel):
    """
    Attention-based model for ECG classification.
    """
    
    def __init__(self, input_shape=(5000, 12), num_classes=5, model_name='ecg_attentionnet'):
        """
        Initialize the ECG AttentionNet model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (samples, leads). Default is (5000, 12).
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_attentionnet'.
        """
        super().__init__(input_shape, num_classes, model_name)
        self.attention_model = None
    
    def _squeeze_excite_block(self, x, ratio=16):
        """
        Create a squeeze-and-excitation block.
        
        Parameters:
        -----------
        x : Tensor
            Input tensor.
        ratio : int
            Reduction ratio. Default is 16.
            
        Returns:
        --------
        x : Tensor
            Output tensor.
        """
        # Get input shape
        channel_axis = -1
        filters = x.shape[channel_axis]
        
        # Squeeze (global average pooling)
        se = GlobalAveragePooling1D()(x)
        
        # Excitation (two dense layers)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        
        # Reshape to match input shape
        se = Reshape((1, filters))(se)
        
        # Scale the input
        x = Multiply()([x, se])
        
        return x
    
    def _attention_block(self, x):
        """
        Create an attention block.
        
        Parameters:
        -----------
        x : Tensor
            Input tensor.
            
        Returns:
        --------
        x : Tensor
            Output tensor.
        attention_weights : Tensor
            Attention weights.
        """
        # Compute attention weights
        attention = Conv1D(1, kernel_size=1)(x)
        attention = Activation('tanh')(attention)
        attention = Flatten()(attention)
        attention_weights = Activation('softmax', name='attention_weights')(attention)
        attention_weights = Reshape((attention_weights.shape[1], 1))(attention_weights)
        
        # Apply attention weights
        x = Multiply()([x, attention_weights])
        
        return x, attention_weights
    
    def build_model(self):
        """
        Build the AttentionNet model architecture.
        
        Returns:
        --------
        model : Model
            Keras model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Convolutional blocks with squeeze-and-excitation
        x = Conv1D(filters=64, kernel_size=5, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self._squeeze_excite_block(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=128, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self._squeeze_excite_block(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self._squeeze_excite_block(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        x, attention_weights = self._attention_block(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Create attention model for visualization
        self.attention_model = Model(
            inputs=inputs,
            outputs=attention_weights,
            name=f"{self.model_name}_attention"
        )
        
        return model
    
    def get_attention_maps(self, X):
        """
        Get attention maps for visualization.
        
        Parameters:
        -----------
        X : array-like
            Input data.
            
        Returns:
        --------
        attention_maps : array-like
            Attention maps.
        """
        if self.attention_model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        return self.attention_model.predict(X)
    
    def plot_attention(self, X, sample_idx=0, lead_idx=0):
        """
        Plot attention weights for a sample.
        
        Parameters:
        -----------
        X : array-like
            Input data.
        sample_idx : int
            Index of the sample to visualize. Default is 0.
        lead_idx : int
            Index of the lead to visualize. Default is 0.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get attention weights
        attention_maps = self.get_attention_maps(X[sample_idx:sample_idx+1])
        attention_weights = attention_maps[0, :, 0]
        
        # Get sample data
        sample = X[sample_idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot ECG signal
        time = np.arange(sample.shape[0]) / 500  # Assuming 500 Hz sampling rate
        ax1.plot(time, sample[:, lead_idx])
        ax1.set_title(f'ECG Signal (Lead {lead_idx})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot attention weights
        ax2.plot(np.linspace(0, time[-1], len(attention_weights)), attention_weights)
        ax2.set_title('Attention Weights')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Weight')
        ax2.grid(True)
        
        fig.tight_layout()
        return fig


class ECGMultiLeadNet(ECGDeepLearningModel):
    """
    Multi-lead ECG classification model.
    """
    
    def __init__(self, input_shape=(5000, 12), num_classes=5, model_name='ecg_multileadnet'):
        """
        Initialize the ECG MultiLeadNet model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (samples, leads). Default is (5000, 12).
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_multileadnet'.
        """
        super().__init__(input_shape, num_classes, model_name)
    
    def _lead_branch(self, inputs, lead_idx):
        """
        Create a branch for a single lead.
        
        Parameters:
        -----------
        inputs : Tensor
            Input tensor.
        lead_idx : int
            Index of the lead.
            
        Returns:
        --------
        x : Tensor
            Output tensor.
        """
        # Extract single lead
        lead = Lambda(lambda x: x[:, :, lead_idx:lead_idx+1], name=f'lead_{lead_idx}')(inputs)
        
        # Convolutional blocks
        x = Conv1D(filters=64, kernel_size=5, padding='same')(lead)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=128, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        return x
    
    def build_model(self):
        """
        Build the MultiLeadNet model architecture.
        
        Returns:
        --------
        model : Model
            Keras model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Create branches for each lead
        lead_features = []
        for i in range(self.input_shape[1]):
            lead_output = self._lead_branch(inputs, i)
            lead_features.append(lead_output)
            
        # Concatenate lead features
        if len(lead_features) > 1:
            x = Concatenate()(lead_features)
        else:
            x = lead_features[0]
            
        # Dense layers
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        return model


class ECGMultiModalNet(ECGDeepLearningModel):
    """
    Multi-modal ECG classification model that combines ECG data with other features.
    """
    
    def __init__(self, ecg_shape=(5000, 12), tabular_shape=10, num_classes=5, model_name='ecg_multimodalnet'):
        """
        Initialize the ECG MultiModalNet model.
        
        Parameters:
        -----------
        ecg_shape : tuple
            Shape of ECG data (samples, leads). Default is (5000, 12).
        tabular_shape : int
            Number of tabular features. Default is 10.
        num_classes : int
            Number of classes. Default is 5.
        model_name : str
            Name of the model. Default is 'ecg_multimodalnet'.
        """
        self.ecg_shape = ecg_shape
        self.tabular_shape = tabular_shape
        super().__init__(ecg_shape, num_classes, model_name)
    
    def build_model(self):
        """
        Build the MultiModalNet model architecture.
        
        Returns:
        --------
        model : Model
            Keras model.
        """
        # ECG input
        ecg_input = Input(shape=self.ecg_shape, name='ecg_input')
        
        # Tabular input
        tabular_input = Input(shape=(self.tabular_shape,), name='tabular_input')
        
        # ECG branch
        x_ecg = Conv1D(filters=64, kernel_size=5, padding='same')(ecg_input)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
        x_ecg = Dropout(0.2)(x_ecg)
        
        x_ecg = Conv1D(filters=128, kernel_size=5, padding='same')(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
        x_ecg = Dropout(0.2)(x_ecg)
        
        x_ecg = Conv1D(filters=256, kernel_size=3, padding='same')(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        x_ecg = Activation('relu')(x_ecg)
        x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
        x_ecg = Dropout(0.2)(x_ecg)
        
        # Global pooling for ECG
        x_ecg = GlobalAveragePooling1D()(x_ecg)
        
        # Tabular branch
        x_tab = Dense(64)(tabular_input)
        x_tab = BatchNormalization()(x_tab)
        x_tab = Activation('relu')(x_tab)
        x_tab = Dropout(0.2)(x_tab)
        
        # Concatenate branches
        x = Concatenate()([x_ecg, x_tab])
        
        # Dense layers
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[ecg_input, tabular_input], outputs=outputs, name=self.model_name)
        
        return model
    
    def fit(self, X, y, validation_data=None, batch_size=32, epochs=50, callbacks=None, verbose=1):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : tuple or list
            Training data (X_ecg, X_tabular).
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
            self.compile_model()
            
        # Unpack inputs
        X_ecg, X_tabular = X
        
        # Convert labels to one-hot encoding
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            self.classes_ = np.unique(y)
            self.num_classes = len(self.classes_)
            
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # Already one-hot encoded
            y_encoded = y
            self.num_classes = y.shape[1]
            self.classes_ = np.arange(self.num_classes)
            
        # Create validation split if not provided
        if validation_data is None:
            # Split ECG data
            X_ecg_train, X_ecg_val, y_train, y_val = train_test_split(
                X_ecg, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Split tabular data
            X_tab_train, X_tab_val = train_test_split(
                X_tabular, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            X_train = [X_ecg_train, X_tab_train]
            X_val = [X_ecg_val, X_tab_val]
        else:
            X_train = [X_ecg, X_tabular]
            (X_ecg_val, X_tab_val), y_val = validation_data
            X_val = [X_ecg_val, X_tab_val]
            y_train = y_encoded
            
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
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : tuple or list
            Features (X_ecg, X_tabular).
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Get predicted probabilities
        y_proba = self.model.predict(X)
        
        # Convert to class labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Map to original classes if available
        if self.classes_ is not None:
            y_pred = self.classes_[y_pred]
            
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : tuple or list
            Features (X_ecg, X_tabular).
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        return self.model.predict(X)
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the model on the given data.
        
        Parameters:
        -----------
        X : tuple or list
            Features (X_ecg, X_tabular).
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
            
        # Convert labels to one-hot if needed
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Get unique classes
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            # Convert to one-hot
            encoder = OneHotEncoder(sparse=False)
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # Already one-hot encoded
            y_encoded = y
            
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_encoded, axis=1)
        
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
            
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
        if 'classification_report' in metrics:
            results['classification_report'] = classification_report(y_true, y_pred)
            
        # Add model evaluation metrics
        model_metrics = self.model.evaluate(X, y_encoded, verbose=0)
        for i, metric_name in enumerate(self.model.metrics_names):
            results[metric_name] = model_metrics[i]
            
        return results


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 100
    n_timesteps = 1000
    n_leads = 12
    n_classes = 5
    
    # ECG data
    X_ecg = np.random.randn(n_samples, n_timesteps, n_leads)
    
    # Tabular data
    X_tabular = np.random.randn(n_samples, 10)
    
    # Labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Create and train ConvNet model
    print("Training ConvNet model...")
    convnet = ECGConvNet(input_shape=(n_timesteps, n_leads), num_classes=n_classes)
    convnet.compile_model()
    convnet.fit(X_ecg, y, epochs=5, batch_size=16, verbose=1)
    
    # Evaluate
    results = convnet.evaluate(X_ecg, y)
    print("\nConvNet evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
            
    # Create and train AttentionNet model
    print("\nTraining AttentionNet model...")
    attentionnet = ECGAttentionNet(input_shape=(n_timesteps, n_leads), num_classes=n_classes)
    attentionnet.compile_model()
    attentionnet.fit(X_ecg, y, epochs=5, batch_size=16, verbose=1)
    
    # Evaluate
    results = attentionnet.evaluate(X_ecg, y)
    print("\nAttentionNet evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
            
    # Create and train MultiModalNet model
    print("\nTraining MultiModalNet model...")
    multimodalnet = ECGMultiModalNet(
        ecg_shape=(n_timesteps, n_leads),
        tabular_shape=10,
        num_classes=n_classes
    )
    multimodalnet.compile_model()
    multimodalnet.fit([X_ecg, X_tabular], y, epochs=5, batch_size=16, verbose=1)
    
    # Evaluate
    results = multimodalnet.evaluate([X_ecg, X_tabular], y)
    print("\nMultiModalNet evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")

