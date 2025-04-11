import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os

class CrocodileClassifier:
    def __init__(self):
        """
        Initialize classification models
        """
        # Initialize models with default parameters
        self.models = {
            'svm': SVC(probability=True, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'xgboost': XGBClassifier(random_state=42)
        }
        
        # Store best model
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
        # Create plots directory if it doesn't exist
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def train_and_evaluate(self, X, y, cv=10):
        """
        Train and evaluate all models using cross-validation
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Evaluation metrics for each model
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv)
            
            # Train model on full dataset
            model.fit(X, y)
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            results[name] = metrics
            
            # Update best model
            if metrics['cv_mean'] > self.best_score:
                self.best_score = metrics['cv_mean']
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def predict(self, X, confidence_threshold=0.5):
        """
        Make predictions using the best model
        
        Args:
            X (numpy.ndarray): Feature matrix
            confidence_threshold (float): Threshold for classification confidence
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Get probability scores
        proba = self.best_model.predict_proba(X)
        
        # Get maximum probability for each prediction
        confidence_scores = np.max(proba, axis=1)
        
        # Make predictions
        predictions = self.best_model.predict(X)
        
        # Set low confidence predictions to "Unknown"
        predictions[confidence_scores < confidence_threshold] = "Unknown"
        
        return predictions, confidence_scores
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """
        Plot confusion matrix
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            labels (list): List of label names
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        
        # Display plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance for models that support it
        
        Args:
            feature_names (list, optional): Names of features
        """
        if self.best_model_name in ['random_forest', 'xgboost']:
            # Get feature importance
            importance = self.best_model.feature_importances_
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            
            # Display plot
            plt.show()
            
            # Close the figure to free memory
            plt.close()
    
    def plot_model_comparison(self, results):
        """
        Plot comparison of different models' performance metrics
        
        Args:
            results (dict): Dictionary containing evaluation metrics for each model
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        # Prepare data for plotting
        metric_values = {metric: [results[model][metric] for model in model_names] 
                        for metric in metrics}
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, metric_values[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Display plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()
    
    def plot_cross_validation_results(self, results):
        """
        Plot cross-validation results with error bars
        
        Args:
            results (dict): Dictionary containing evaluation metrics for each model
        """
        model_names = list(results.keys())
        cv_means = [results[model]['cv_mean'] for model in model_names]
        cv_stds = [results[model]['cv_std'] for model in model_names]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(model_names, cv_means, yerr=cv_stds, fmt='o', capsize=5)
        plt.title('Cross-validation Results')
        plt.xlabel('Models')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
        
        # Display plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()
    
    def plot_roc_curves(self, X, y):
        """
        Plot ROC curves for each class
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
        """
        # Binarize the labels for ROC curve calculation
        classes = np.unique(y)
        y_bin = label_binarize(y, classes=classes)
        n_classes = y_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            proba = self.best_model.predict_proba(X)[:, i]
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve (class {classes[i]}, AUC = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        
        # Display plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()
    
    def plot_confidence_distribution(self, confidence_scores, predictions, threshold=0.5, filename='confidence_distribution.png'):
        """
        Plot distribution of confidence scores
        
        Args:
            confidence_scores (numpy.ndarray): Confidence scores from predictions
            predictions (numpy.ndarray): Predicted labels
            threshold (float): Confidence threshold used for classification
            filename (str): Name of the file to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of confidence scores
        plt.hist(confidence_scores, bins=30, alpha=0.7)
        
        # Add vertical line for threshold
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Threshold ({threshold})')
        
        # Add text with statistics
        known_count = np.sum(confidence_scores >= threshold)
        unknown_count = np.sum(confidence_scores < threshold)
        total = len(confidence_scores)
        
        stats_text = f'Known: {known_count}/{total} ({known_count/total*100:.1f}%)\n'
        stats_text += f'Unknown: {unknown_count}/{total} ({unknown_count/total*100:.1f}%)'
        
        plt.text(0.98, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        
        # Display plot
        plt.show()
        
        # Close the figure to free memory
        plt.close() 