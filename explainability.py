"""
CardioInsight AI - Explainability Module

This module provides tools for explaining and visualizing AI model decisions.
It includes Grad-CAM, attention visualization, and other explainability techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns

class GradCAM:
    """
    Class for generating Grad-CAM visualizations for ECG models.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize the Grad-CAM visualizer.
        
        Parameters:
        -----------
        model : Model
            Keras model.
        layer_name : str or None
            Name of the layer to use for Grad-CAM. If None, uses the last convolutional layer.
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
                    
        if layer_name is None:
            raise ValueError("Could not find a convolutional layer in the model.")
            
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()
        
    def _create_grad_model(self):
        """
        Create a model that outputs both the predictions and the activations of the target layer.
        
        Returns:
        --------
        grad_model : Model
            Keras model for computing gradients.
        """
        # Get the target layer
        target_layer = self.model.get_layer(self.layer_name)
        
        # Create a model that maps the input to the activations of the target layer and the output
        return Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )
    
    def compute_heatmap(self, ecg_data, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap for the given ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data with shape matching the model input.
        class_idx : int or None
            Index of the class for which to compute the heatmap.
            If None, uses the predicted class.
        eps : float
            Small constant to avoid division by zero.
            
        Returns:
        --------
        heatmap : array-like
            Grad-CAM heatmap.
        """
        # Ensure data is in batch format
        if len(ecg_data.shape) == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        # Get the predicted class if not specified
        if class_idx is None:
            predictions = self.model.predict(ecg_data)
            class_idx = np.argmax(predictions[0])
            
        # Compute gradients and activations
        with tf.GradientTape() as tape:
            # Cast inputs to float32
            if isinstance(ecg_data, list):
                inputs = [tf.cast(x, tf.float32) for x in ecg_data]
            else:
                inputs = tf.cast(ecg_data, tf.float32)
                
            # Get activations and predictions
            activations, predictions = self.grad_model(inputs)
            
            # Get loss for the target class
            loss = predictions[:, class_idx]
            
        # Compute gradients with respect to activations
        grads = tape.gradient(loss, activations)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Weight the activations by the gradients
        activations = activations[0]
        weighted_activations = tf.multiply(activations, pooled_grads)
        
        # Average over all filters to get heatmap
        heatmap = tf.reduce_sum(weighted_activations, axis=-1).numpy()
        
        # ReLU to only keep positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, ecg_data, heatmap, lead_idx=0, alpha=0.4):
        """
        Overlay heatmap on ECG signal.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        heatmap : array-like
            Grad-CAM heatmap.
        lead_idx : int
            Index of the lead to visualize. Default is 0.
        alpha : float
            Transparency of the heatmap. Default is 0.4.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Extract the specified lead
        if len(ecg_data.shape) == 3:
            # Batch data, take first sample
            lead_data = ecg_data[0, :, lead_idx]
        elif len(ecg_data.shape) == 2:
            # Single sample
            lead_data = ecg_data[:, lead_idx]
        else:
            # Already a single lead
            lead_data = ecg_data
            
        # Resize heatmap to match ECG length if needed
        if len(heatmap) != len(lead_data):
            heatmap = cv2.resize(heatmap, (1, len(lead_data)))
            heatmap = heatmap.flatten()
            
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot ECG signal
        time = np.arange(len(lead_data))
        ax.plot(time, lead_data, 'k', alpha=0.8)
        
        # Create colormap for heatmap
        cmap = plt.cm.jet
        
        # Plot heatmap as colored background
        ax.fill_between(
            time, lead_data.min(), lead_data.max(),
            color='blue', alpha=0.0  # Start with transparent
        )
        
        # Add colored regions based on heatmap
        for i in range(len(time) - 1):
            ax.fill_between(
                [time[i], time[i+1]], lead_data.min(), lead_data.max(),
                color=cmap(heatmap[i]), alpha=alpha
            )
            
        ax.set_title(f'ECG Signal with Grad-CAM Heatmap (Lead {lead_idx})')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(heatmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Activation')
        
        fig.tight_layout()
        return fig
    
    def visualize_multiple_leads(self, ecg_data, class_idx=None, leads_to_show=None, figsize=(15, 10)):
        """
        Visualize Grad-CAM heatmaps for multiple leads.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        class_idx : int or None
            Index of the class for which to compute the heatmap.
            If None, uses the predicted class.
        leads_to_show : list or None
            List of lead indices to show. If None, shows all leads.
        figsize : tuple
            Figure size. Default is (15, 10).
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Ensure data is in batch format
        if len(ecg_data.shape) == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        # Get number of leads
        n_leads = ecg_data.shape[2]
        
        # Set leads to show
        if leads_to_show is None:
            leads_to_show = list(range(n_leads))
            
        n_leads_to_show = len(leads_to_show)
        
        # Compute heatmap
        heatmap = self.compute_heatmap(ecg_data, class_idx)
        
        # Create figure
        n_rows = int(np.ceil(n_leads_to_show / 3))
        n_cols = min(n_leads_to_show, 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each lead
        for i, lead_idx in enumerate(leads_to_show):
            if i >= len(axes):
                break
                
            # Extract lead data
            lead_data = ecg_data[0, :, lead_idx]
            
            # Resize heatmap to match ECG length if needed
            if len(heatmap) != len(lead_data):
                lead_heatmap = cv2.resize(heatmap, (1, len(lead_data)))
                lead_heatmap = lead_heatmap.flatten()
            else:
                lead_heatmap = heatmap
                
            # Plot ECG signal
            time = np.arange(len(lead_data))
            axes[i].plot(time, lead_data, 'k', alpha=0.8)
            
            # Create colormap for heatmap
            cmap = plt.cm.jet
            
            # Plot heatmap as colored background
            axes[i].fill_between(
                time, lead_data.min(), lead_data.max(),
                color='blue', alpha=0.0  # Start with transparent
            )
            
            # Add colored regions based on heatmap
            for j in range(len(time) - 1):
                axes[i].fill_between(
                    [time[j], time[j+1]], lead_data.min(), lead_data.max(),
                    color=cmap(lead_heatmap[j]), alpha=0.4
                )
                
            axes[i].set_title(f'Lead {lead_idx}')
            axes[i].grid(True, alpha=0.3)
            
            # Remove x and y ticks for cleaner look
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
        # Hide unused subplots
        for i in range(n_leads_to_show, len(axes)):
            axes[i].set_visible(False)
            
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(heatmap)
        cbar = fig.colorbar(sm, ax=axes.tolist())
        cbar.set_label('Activation')
        
        # Get predicted class if not specified
        if class_idx is None:
            predictions = self.model.predict(ecg_data)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0, class_idx]
            fig.suptitle(f'Grad-CAM Visualization for Class {class_idx} (Confidence: {confidence:.2f})', fontsize=16)
        else:
            fig.suptitle(f'Grad-CAM Visualization for Class {class_idx}', fontsize=16)
            
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        return fig


class AttentionVisualizer:
    """
    Class for visualizing attention weights in ECG models.
    """
    
    def __init__(self, model):
        """
        Initialize the attention visualizer.
        
        Parameters:
        -----------
        model : Model
            Keras model with attention mechanism.
        """
        self.model = model
        
    def get_attention_weights(self, ecg_data):
        """
        Get attention weights for the given ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
            
        Returns:
        --------
        attention_weights : array-like
            Attention weights.
        """
        # Check if model has get_attention_maps method
        if hasattr(self.model, 'get_attention_maps'):
            return self.model.get_attention_maps(ecg_data)
            
        # Try to find attention layer
        attention_layer = None
        for layer in self.model.layers:
            if 'attention' in layer.name.lower():
                attention_layer = layer
                break
                
        if attention_layer is None:
            raise ValueError("Could not find attention layer in the model.")
            
        # Create a model that outputs attention weights
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=attention_layer.output
        )
        
        # Get attention weights
        return attention_model.predict(ecg_data)
    
    def visualize_attention(self, ecg_data, lead_idx=0, sample_idx=0):
        """
        Visualize attention weights for a sample.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        lead_idx : int
            Index of the lead to visualize. Default is 0.
        sample_idx : int
            Index of the sample to visualize. Default is 0.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Ensure data is in batch format
        if len(ecg_data.shape) == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        # Extract sample
        sample = ecg_data[sample_idx:sample_idx+1]
        
        # Get attention weights
        attention_weights = self.get_attention_weights(sample)
        
        # Extract weights for the sample
        if len(attention_weights.shape) == 3:
            # Shape: (batch_size, time_steps, 1)
            weights = attention_weights[0, :, 0]
        else:
            # Shape: (batch_size, time_steps)
            weights = attention_weights[0]
            
        # Extract lead data
        lead_data = sample[0, :, lead_idx]
        
        # Resize weights to match ECG length if needed
        if len(weights) != len(lead_data):
            weights = cv2.resize(weights.reshape(-1, 1), (1, len(lead_data)))
            weights = weights.flatten()
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot ECG signal
        time = np.arange(len(lead_data))
        ax1.plot(time, lead_data)
        ax1.set_title(f'ECG Signal (Lead {lead_idx})')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot attention weights
        ax2.plot(time, weights)
        ax2.set_title('Attention Weights')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Weight')
        ax2.grid(True)
        
        fig.tight_layout()
        return fig
    
    def visualize_attention_heatmap(self, ecg_data, lead_idx=0, sample_idx=0, alpha=0.4):
        """
        Visualize attention weights as a heatmap overlaid on the ECG signal.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
        lead_idx : int
            Index of the lead to visualize. Default is 0.
        sample_idx : int
            Index of the sample to visualize. Default is 0.
        alpha : float
            Transparency of the heatmap. Default is 0.4.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Ensure data is in batch format
        if len(ecg_data.shape) == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        # Extract sample
        sample = ecg_data[sample_idx:sample_idx+1]
        
        # Get attention weights
        attention_weights = self.get_attention_weights(sample)
        
        # Extract weights for the sample
        if len(attention_weights.shape) == 3:
            # Shape: (batch_size, time_steps, 1)
            weights = attention_weights[0, :, 0]
        else:
            # Shape: (batch_size, time_steps)
            weights = attention_weights[0]
            
        # Extract lead data
        lead_data = sample[0, :, lead_idx]
        
        # Resize weights to match ECG length if needed
        if len(weights) != len(lead_data):
            weights = cv2.resize(weights.reshape(-1, 1), (1, len(lead_data)))
            weights = weights.flatten()
            
        # Normalize weights
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot ECG signal
        time = np.arange(len(lead_data))
        ax.plot(time, lead_data, 'k', alpha=0.8)
        
        # Create colormap for heatmap
        cmap = plt.cm.jet
        
        # Plot heatmap as colored background
        ax.fill_between(
            time, lead_data.min(), lead_data.max(),
            color='blue', alpha=0.0  # Start with transparent
        )
        
        # Add colored regions based on attention weights
        for i in range(len(time) - 1):
            ax.fill_between(
                [time[i], time[i+1]], lead_data.min(), lead_data.max(),
                color=cmap(weights[i]), alpha=alpha
            )
            
        ax.set_title(f'ECG Signal with Attention Heatmap (Lead {lead_idx})')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(weights)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight')
        
        fig.tight_layout()
        return fig


class ConfidenceVisualizer:
    """
    Class for visualizing model confidence and uncertainty.
    """
    
    def __init__(self, model):
        """
        Initialize the confidence visualizer.
        
        Parameters:
        -----------
        model : Model
            Keras model or any model with predict_proba method.
        """
        self.model = model
        
    def get_confidence_scores(self, X):
        """
        Get confidence scores for predictions.
        
        Parameters:
        -----------
        X : array-like
            Input data.
            
        Returns:
        --------
        confidence : array-like
            Confidence scores.
        predictions : array-like
            Predicted classes.
        """
        # Get predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = self.model.predict(X)
            
        # Get predicted classes and confidence scores
        predictions = np.argmax(y_proba, axis=1)
        confidence = np.max(y_proba, axis=1)
        
        return confidence, predictions
    
    def plot_confidence_histogram(self, X, y=None, bins=10):
        """
        Plot histogram of confidence scores.
        
        Parameters:
        -----------
        X : array-like
            Input data.
        y : array-like or None
            True labels. If provided, colors correct and incorrect predictions differently.
        bins : int
            Number of bins. Default is 10.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get confidence scores and predictions
        confidence, predictions = self.get_confidence_scores(X)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if y is not None:
            # Separate correct and incorrect predictions
            correct = predictions == y
            
            # Plot histograms
            ax.hist(confidence[correct], bins=bins, alpha=0.7, label='Correct predictions')
            ax.hist(confidence[~correct], bins=bins, alpha=0.7, label='Incorrect predictions')
            ax.legend()
        else:
            # Plot single histogram
            ax.hist(confidence, bins=bins)
            
        ax.set_title('Confidence Score Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def plot_confidence_vs_accuracy(self, X, y, bins=10):
        """
        Plot confidence vs. accuracy calibration curve.
        
        Parameters:
        -----------
        X : array-like
            Input data.
        y : array-like
            True labels.
        bins : int
            Number of bins. Default is 10.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get confidence scores and predictions
        confidence, predictions = self.get_confidence_scores(X)
        
        # Compute accuracy per confidence bin
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(confidence, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(predictions[mask] == y[mask])
                bin_conf = np.mean(confidence[mask])
                bin_count = np.sum(mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
                
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot calibration curve
        ax.plot(bin_confidences, bin_accuracies, 'o-', label='Calibration curve')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot confidence histogram
        ax2 = ax.twinx()
        ax2.bar(bin_confidences, bin_counts, alpha=0.2, width=0.05, label='Sample count')
        ax2.set_ylabel('Count')
        
        ax.set_title('Confidence vs. Accuracy Calibration Curve')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        fig.tight_layout()
        return fig
    
    def plot_roc_curves(self, X, y):
        """
        Plot ROC curves for each class.
        
        Parameters:
        -----------
        X : array-like
            Input data.
        y : array-like
            True labels.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = self.model.predict(X)
            
        # Get number of classes
        n_classes = y_proba.shape[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            # Binarize labels
            y_bin = (y == i).astype(int)
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        ax.set_title('ROC Curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        fig.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, X, y, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        X : array-like
            Input data.
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
        from sklearn.metrics import confusion_matrix
        
        # Get predictions
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
        else:
            _, y_pred = self.get_confidence_scores(X)
            
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = np.unique(y)
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
            title = 'Confusion Matrix'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig


class FeatureImportanceVisualizer:
    """
    Class for visualizing feature importance in ECG models.
    """
    
    def __init__(self, model):
        """
        Initialize the feature importance visualizer.
        
        Parameters:
        -----------
        model : Model
            Model with feature_importances_ attribute or coefficients.
        """
        self.model = model
        
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance.
        
        Parameters:
        -----------
        feature_names : list or None
            List of feature names. If None, uses feature indices.
            
        Returns:
        --------
        importance : array-like
            Feature importance scores.
        names : list
            Feature names.
        """
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        # Check if model has coef_ attribute (linear models)
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
        # Try to access feature importances through model.model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importances_'):
            importance = self.model.model.feature_importances_
        # Try to access coefficients through model.model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'coef_'):
            importance = np.abs(self.model.model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
        else:
            raise ValueError("Could not find feature importances in the model.")
            
        # Set feature names
        if feature_names is None:
            names = [f"Feature {i}" for i in range(len(importance))]
        else:
            names = feature_names
            
        return importance, names
    
    def plot_feature_importance(self, feature_names=None, top_n=20, title=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_names : list or None
            List of feature names. If None, uses feature indices.
        top_n : int
            Number of top features to show. Default is 20.
        title : str or None
            Plot title. If None, uses a default title.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get feature importance
        importance, names = self.get_feature_importance(feature_names)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        # Limit to top_n features
        indices = indices[:top_n]
        top_importance = importance[indices]
        top_names = [names[i] for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(len(top_importance)), top_importance)
        ax.set_xticks(range(len(top_importance)))
        ax.set_xticklabels(top_names, rotation=90)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        
        # Set title
        if title is None:
            title = 'Feature Importance'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig
    
    def plot_feature_importance_heatmap(self, feature_names=None, top_n=20, title=None):
        """
        Plot feature importance as a heatmap.
        
        Parameters:
        -----------
        feature_names : list or None
            List of feature names. If None, uses feature indices.
        top_n : int
            Number of top features to show. Default is 20.
        title : str or None
            Plot title. If None, uses a default title.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get feature importance
        importance, names = self.get_feature_importance(feature_names)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        # Limit to top_n features
        indices = indices[:top_n]
        top_importance = importance[indices]
        top_names = [names[i] for i in indices]
        
        # Create DataFrame for heatmap
        df = pd.DataFrame({'Feature': top_names, 'Importance': top_importance})
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            df.set_index('Feature').T,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            ax=ax
        )
        
        # Set title
        if title is None:
            title = 'Feature Importance Heatmap'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Import necessary modules
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
    
    # Generate synthetic data
    n_samples = 10
    n_timesteps = 1000
    n_leads = 12
    n_classes = 5
    
    X = np.random.randn(n_samples, n_timesteps, n_leads)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Create a simple CNN model
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_leads)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model (just for demonstration)
    model.fit(X, y, epochs=1, batch_size=2, verbose=0)
    
    # Create Grad-CAM visualizer
    gradcam = GradCAM(model)
    
    # Compute and visualize heatmap for a sample
    sample_idx = 0
    lead_idx = 0
    
    heatmap = gradcam.compute_heatmap(X[sample_idx:sample_idx+1])
    fig = gradcam.overlay_heatmap(X[sample_idx], heatmap, lead_idx)
    
    print("Grad-CAM visualization created.")
    
    # Create confidence visualizer
    conf_viz = ConfidenceVisualizer(model)
    
    # Get confidence scores
    confidence, predictions = conf_viz.get_confidence_scores(X)
    
    print("\nConfidence scores:")
    for i in range(5):
        print(f"Sample {i}: Predicted class {predictions[i]} with confidence {confidence[i]:.4f}")
        
    print("\nFeature importance visualization is not applicable for this CNN model.")

