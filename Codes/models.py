import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        plt.show()
    
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
            plt.show() 