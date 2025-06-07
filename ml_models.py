"""
CardioInsight AI - Machine Learning Models Module

This module provides machine learning models for ECG classification.
It includes traditional ML models and evaluation functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
import os
import time

class ECGClassifier:
    """
    Class for training and evaluating ECG classification models.
    """
    
    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initialize the ECG classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options: 'random_forest', 'svm', 'gradient_boosting',
            'knn', 'logistic_regression', 'mlp'. Default is 'random_forest'.
        model_params : dict or None
            Parameters for the model. If None, uses default parameters.
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.classes_ = None
        self.feature_importances_ = None
        
    def _create_model(self):
        """
        Create a model based on the specified type.
        
        Returns:
        --------
        model : estimator
            Scikit-learn estimator.
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                random_state=42
            )
        
        elif self.model_type == 'svm':
            return SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                gamma=self.model_params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 3),
                random_state=42
            )
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.model_params.get('n_neighbors', 5),
                weights=self.model_params.get('weights', 'uniform'),
                algorithm=self.model_params.get('algorithm', 'auto')
            )
        
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.model_params.get('C', 1.0),
                penalty=self.model_params.get('penalty', 'l2'),
                solver=self.model_params.get('solver', 'lbfgs'),
                max_iter=self.model_params.get('max_iter', 1000),
                random_state=42
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=self.model_params.get('hidden_layer_sizes', (100,)),
                activation=self.model_params.get('activation', 'relu'),
                solver=self.model_params.get('solver', 'adam'),
                alpha=self.model_params.get('alpha', 0.0001),
                learning_rate=self.model_params.get('learning_rate', 'constant'),
                max_iter=self.model_params.get('max_iter', 200),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Store classes
        self.classes_ = self.model.classes_
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
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
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return self.model.predict(X_scaled)
    
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
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        return self.model.predict_proba(X_scaled)
    
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
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        
        # Compute metrics
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, y_pred)
            
        if 'precision' in metrics:
            results['precision'] = precision_score(y, y_pred, average='weighted')
            
        if 'recall' in metrics:
            results['recall'] = recall_score(y, y_pred, average='weighted')
            
        if 'f1' in metrics:
            results['f1'] = f1_score(y, y_pred, average='weighted')
            
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y, y_pred)
            
        if 'classification_report' in metrics:
            results['classification_report'] = classification_report(y, y_pred)
            
        return results
    
    def cross_validate(self, X, y, cv=5, metrics=None):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
        cv : int
            Number of folds. Default is 5.
        metrics : list or None
            List of metrics to compute. If None, computes accuracy, precision, recall, and F1.
            
        Returns:
        --------
        results : dict
            Dictionary of cross-validation results.
        """
        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute cross-validation scores
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='precision_weighted')
            elif metric == 'recall':
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='recall_weighted')
            elif metric == 'f1':
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='f1_weighted')
            else:
                continue
                
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_scores'] = scores
            
        return results
    
    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='accuracy'):
        """
        Tune hyperparameters using grid search.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
        param_grid : dict
            Dictionary of parameters to search.
        cv : int
            Number of folds. Default is 5.
        scoring : str
            Scoring metric. Default is 'accuracy'.
            
        Returns:
        --------
        self : object
            Fitted estimator with best parameters.
        best_params : dict
            Best parameters found.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        # Store classes
        self.classes_ = self.model.classes_
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        return self, grid_search.best_params_
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'classes': self.classes_,
            'feature_importances': self.feature_importances_
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        classifier : ECGClassifier
            Loaded classifier.
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create classifier
        classifier = cls(
            model_type=model_data['model_type'],
            model_params=model_data['model_params']
        )
        
        # Set model attributes
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.classes_ = model_data['classes']
        classifier.feature_importances_ = model_data['feature_importances']
        
        return classifier
    
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
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        
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
        classes = self.classes_
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
            title = f'Confusion Matrix ({self.model_type})'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig
    
    def plot_roc_curve(self, X, y, title=None):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            True labels.
        title : str or None
            Plot title. If None, uses a default title.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Binarize labels for multi-class ROC
        classes = self.classes_
        n_classes = len(classes)
        
        if n_classes == 2:
            # Binary classification
            y_score = self.model.predict_proba(X_scaled)[:, 1]
            
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            
            # Set title
            if title is None:
                title = f'ROC Curve ({self.model_type})'
            ax.set_title(title)
            
            ax.legend(loc="lower right")
            
        else:
            # Multi-class classification
            y_bin = label_binarize(y, classes=classes)
            y_score = self.model.predict_proba(X_scaled)
            
            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            # Plot ROC curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i in range(n_classes):
                ax.plot(
                    fpr[i], tpr[i], lw=2,
                    label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})'
                )
                
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            
            # Set title
            if title is None:
                title = f'Multi-class ROC Curve ({self.model_type})'
            ax.set_title(title)
            
            ax.legend(loc="lower right")
            
        fig.tight_layout()
        return fig
    
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
        fig : Figure or None
            Matplotlib figure, or None if feature importances are not available.
        """
        # Check if feature importances are available
        if self.feature_importances_ is None:
            print("Feature importances not available for this model.")
            return None
            
        # Get feature importances
        importances = self.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Limit to top_n features
        indices = indices[:top_n]
        top_importances = importances[indices]
        
        # Set feature names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
        top_names = [feature_names[i] for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(len(top_importances)), top_importances)
        ax.set_xticks(range(len(top_importances)))
        ax.set_xticklabels(top_names, rotation=90)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        
        # Set title
        if title is None:
            title = f'Feature Importance ({self.model_type})'
        ax.set_title(title)
        
        fig.tight_layout()
        return fig


class ECGMultiModelClassifier:
    """
    Class for training and evaluating multiple ECG classification models.
    """
    
    def __init__(self, models=None):
        """
        Initialize the multi-model classifier.
        
        Parameters:
        -----------
        models : list or None
            List of model configurations. Each configuration is a dict with 'type' and 'params' keys.
            If None, uses default models.
        """
        if models is None:
            # Default models
            models = [
                {'type': 'random_forest', 'params': {'n_estimators': 100}},
                {'type': 'svm', 'params': {'C': 1.0, 'kernel': 'rbf'}},
                {'type': 'gradient_boosting', 'params': {'n_estimators': 100}},
                {'type': 'knn', 'params': {'n_neighbors': 5}},
                {'type': 'logistic_regression', 'params': {'C': 1.0}}
            ]
            
        # Create classifiers
        self.classifiers = {}
        for model_config in models:
            model_type = model_config['type']
            model_params = model_config.get('params', {})
            self.classifiers[model_type] = ECGClassifier(model_type, model_params)
            
        self.best_model = None
        self.best_model_type = None
        self.results = {}
        
    def fit_all(self, X, y):
        """
        Fit all models to the data.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        for model_type, classifier in self.classifiers.items():
            print(f"Training {model_type} model...")
            start_time = time.time()
            classifier.fit(X, y)
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
            
        return self
    
    def evaluate_all(self, X, y, metrics=None):
        """
        Evaluate all models on the given data.
        
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
            Dictionary of evaluation results for all models.
        """
        results = {}
        
        for model_type, classifier in self.classifiers.items():
            print(f"Evaluating {model_type} model...")
            model_results = classifier.evaluate(X, y, metrics)
            results[model_type] = model_results
            
        self.results = results
        return results
    
    def cross_validate_all(self, X, y, cv=5, metrics=None):
        """
        Perform cross-validation for all models.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
        cv : int
            Number of folds. Default is 5.
        metrics : list or None
            List of metrics to compute. If None, computes accuracy, precision, recall, and F1.
            
        Returns:
        --------
        results : dict
            Dictionary of cross-validation results for all models.
        """
        results = {}
        
        for model_type, classifier in self.classifiers.items():
            print(f"Cross-validating {model_type} model...")
            model_results = classifier.cross_validate(X, y, cv, metrics)
            results[model_type] = model_results
            
        self.results = results
        return results
    
    def select_best_model(self, metric='accuracy_mean'):
        """
        Select the best model based on the specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for selection. Default is 'accuracy_mean'.
            
        Returns:
        --------
        best_model : ECGClassifier
            Best model.
        best_model_type : str
            Type of the best model.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all() or cross_validate_all() first.")
            
        # Find best model
        best_score = -float('inf')
        best_model_type = None
        
        for model_type, model_results in self.results.items():
            if metric in model_results and model_results[metric] > best_score:
                best_score = model_results[metric]
                best_model_type = model_type
                
        if best_model_type is None:
            raise ValueError(f"Metric '{metric}' not found in evaluation results.")
            
        self.best_model = self.classifiers[best_model_type]
        self.best_model_type = best_model_type
        
        return self.best_model, best_model_type
    
    def predict(self, X, model_type=None):
        """
        Predict class labels using the specified model or the best model.
        
        Parameters:
        -----------
        X : array-like
            Features.
        model_type : str or None
            Type of model to use. If None, uses the best model if available,
            otherwise raises an error.
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels.
        """
        if model_type is not None:
            if model_type not in self.classifiers:
                raise ValueError(f"Model type '{model_type}' not found.")
            return self.classifiers[model_type].predict(X)
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
            
        return self.best_model.predict(X)
    
    def predict_proba(self, X, model_type=None):
        """
        Predict class probabilities using the specified model or the best model.
        
        Parameters:
        -----------
        X : array-like
            Features.
        model_type : str or None
            Type of model to use. If None, uses the best model if available,
            otherwise raises an error.
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities.
        """
        if model_type is not None:
            if model_type not in self.classifiers:
                raise ValueError(f"Model type '{model_type}' not found.")
            return self.classifiers[model_type].predict_proba(X)
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
            
        return self.best_model.predict_proba(X)
    
    def save_models(self, directory):
        """
        Save all models to files.
        
        Parameters:
        -----------
        directory : str
            Directory to save the models.
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        for model_type, classifier in self.classifiers.items():
            filepath = os.path.join(directory, f"{model_type}_model.joblib")
            classifier.save_model(filepath)
            
        # Save best model info if available
        if self.best_model_type is not None:
            with open(os.path.join(directory, "best_model_info.txt"), "w") as f:
                f.write(f"Best model type: {self.best_model_type}\n")
                
                if self.results and self.best_model_type in self.results:
                    f.write("Evaluation results:\n")
                    for metric, value in self.results[self.best_model_type].items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
    
    @classmethod
    def load_models(cls, directory):
        """
        Load models from files.
        
        Parameters:
        -----------
        directory : str
            Directory containing the saved models.
            
        Returns:
        --------
        multi_classifier : ECGMultiModelClassifier
            Loaded multi-model classifier.
        """
        # Create empty multi-model classifier
        multi_classifier = cls(models=[])
        
        # Load each model
        for filename in os.listdir(directory):
            if filename.endswith("_model.joblib"):
                model_type = filename.split("_model.joblib")[0]
                filepath = os.path.join(directory, filename)
                classifier = ECGClassifier.load_model(filepath)
                multi_classifier.classifiers[model_type] = classifier
                
        # Load best model info if available
        best_model_info_path = os.path.join(directory, "best_model_info.txt")
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, "r") as f:
                for line in f:
                    if line.startswith("Best model type:"):
                        best_model_type = line.split("Best model type:")[1].strip()
                        multi_classifier.best_model_type = best_model_type
                        multi_classifier.best_model = multi_classifier.classifiers.get(best_model_type)
                        break
                        
        return multi_classifier
    
    def plot_model_comparison(self, metric='accuracy_mean', title=None):
        """
        Plot model comparison based on the specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison. Default is 'accuracy_mean'.
        title : str or None
            Plot title. If None, uses a default title.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all() or cross_validate_all() first.")
            
        # Extract metric values for each model
        model_types = []
        metric_values = []
        
        for model_type, model_results in self.results.items():
            if metric in model_results:
                model_types.append(model_type)
                metric_values.append(model_results[metric])
                
        if not model_types:
            raise ValueError(f"Metric '{metric}' not found in evaluation results.")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(model_types, metric_values)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        
        # Set title
        if title is None:
            title = f'Model Comparison ({metric})'
        ax.set_title(title)
        
        # Add values on top of bars
        for i, v in enumerate(metric_values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
            
        fig.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Import necessary modules
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a single classifier
    print("Training a single classifier...")
    classifier = ECGClassifier(model_type='random_forest')
    classifier.fit(X_train, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    print("Evaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        
    # Cross-validate
    cv_results = classifier.cross_validate(X, y)
    print("\nCross-validation results:")
    for metric, value in cv_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
            
    # Create and train multiple classifiers
    print("\nTraining multiple classifiers...")
    multi_classifier = ECGMultiModelClassifier()
    multi_classifier.fit_all(X_train, y_train)
    
    # Cross-validate all models
    cv_results = multi_classifier.cross_validate_all(X, y)
    print("\nCross-validation results for all models:")
    for model_type, model_results in cv_results.items():
        print(f"\n{model_type}:")
        for metric, value in model_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value}")
                
    # Select best model
    best_model, best_model_type = multi_classifier.select_best_model()
    print(f"\nBest model: {best_model_type}")
    
    # Predict using best model
    y_pred = multi_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best model accuracy: {accuracy:.4f}")

