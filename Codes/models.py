import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
from utils import create_directory
import xgboost as xgb

class CrocodileClassifier:
    def __init__(self):
        """
        Initialize the crocodile classifier with memory-efficient components
        """
        # Initialize scaler with memory-efficient settings
        self.scaler = StandardScaler()
        
        # Initialize PCA with consistent components and batch size
        self.pca = IncrementalPCA(n_components=2500, batch_size=2500)
        
        # Initialize models with optimized parameters
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        }
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        self.cache_dir = 'model_cache'
        create_directory(self.cache_dir)
    
    def preprocess_features(self, features, is_training=True):
        """
        Preprocess features with memory-efficient batch processing
        
        Args:
            features (np.ndarray): Input features
            is_training (bool): Whether this is training data
            
        Returns:
            np.ndarray: Preprocessed features
        """
        print("\n=== Feature Preprocessing ===")
        print(f"Original feature dimension: {features.shape[1]}")
        
        # Process in smaller batches to reduce memory usage
        batch_size = 1000
        n_samples = features.shape[0]
        processed_features = []
        
        # Scale features in batches
        print("Scaling features...")
        for i in tqdm(range(0, n_samples, batch_size)):
            batch = features[i:i+batch_size]
            if is_training:
                scaled_batch = self.scaler.fit_transform(batch)
            else:
                scaled_batch = self.scaler.transform(batch)
            processed_features.append(scaled_batch)
        
        features = np.vstack(processed_features)
        print(f"Features scaled. Memory usage reduced by {features.nbytes / (1024**2):.2f} MB")
        
        # Apply PCA in batches
        print("Applying PCA...")
        if is_training:
            # Fit PCA on a subset of data
            subset_size = min(10000, n_samples)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
            self.pca.fit(features[subset_indices])
            print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Transform in batches
        processed_features = []
        for i in tqdm(range(0, n_samples, batch_size)):
            batch = features[i:i+batch_size]
            reduced_batch = self.pca.transform(batch)
            processed_features.append(reduced_batch)
        
        features = np.vstack(processed_features)
        print(f"PCA applied. Reduced dimension: {features.shape[1]}")
        print(f"Memory usage reduced by {features.nbytes / (1024**2):.2f} MB")
        print("===========================\n")
        
        return features
    
    def train_and_evaluate(self, X, y):
        """
        Train and evaluate models with memory-efficient processing
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            dict: Evaluation results
        """
        print("\n=== Model Training and Evaluation ===")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, is_training=True)
        
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_processed, y)
            
            # Evaluate with cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_processed, y, cv=cv, n_jobs=-1)
            
            # Make predictions
            y_pred = model.predict(X_processed)
            y_prob = model.predict_proba(X_processed)
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y, y_prob, multi_class='ovr'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Completed {name}")
        
        print("\n=== Training Complete ===\n")
        return results
    
    def predict(self, X):
        """
        Make predictions with memory-efficient processing
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            tuple: (predictions, confidence scores)
        """
        # Preprocess features
        X_processed = self.preprocess_features(X, is_training=False)
        
        # Get predictions from all models
        predictions = []
        confidences = []
        
        for model in self.models.values():
            pred = model.predict(X_processed)
            prob = model.predict_proba(X_processed)
            confidence = np.max(prob, axis=1)
            
            predictions.append(pred)
            confidences.append(confidence)
        
        # Use ensemble voting
        final_predictions = []
        final_confidence = []
        
        for i in range(len(X)):
            votes = {}
            for j, pred in enumerate(predictions):
                if pred[i] not in votes:
                    votes[pred[i]] = []
                votes[pred[i]].append(confidences[j][i])
            
            # Get prediction with highest average confidence
            best_pred = max(votes.items(), key=lambda x: np.mean(x[1]))[0]
            avg_confidence = np.mean(votes[best_pred])
            
            final_predictions.append(best_pred)
            final_confidence.append(avg_confidence)
        
        return np.array(final_predictions), np.array(final_confidence)
    
    def plot_model_comparison(self, results):
        """Plot model comparison visualization with improved formatting"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(metrics))
        width = 0.35  # Slightly wider bars
        
        # Create larger figure with better proportions
        plt.figure(figsize=(14, 8))
        
        # Use a more attractive color palette
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        # Plot bars with better styling
        for i, (name, scores) in enumerate(results.items()):
            values = [scores.get(m, 0) for m in metrics]  # Handle missing metrics
            plt.bar(
                x + i*width, 
                values, 
                width, 
                label=name,
                color=colors[i % len(colors)],
                edgecolor='white',
                linewidth=1,
                alpha=0.8
            )
        
        # Add value labels on top of bars
        for i, (name, scores) in enumerate(results.items()):
            for j, m in enumerate(metrics):
                if m in scores:
                    plt.text(
                        x[j] + i*width, 
                        scores[m] + 0.01, 
                        f'{scores[m]:.2f}', 
                        ha='center', 
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )
        
        # Improve styling and readability
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width/2, [m.capitalize() for m in metrics], fontsize=10)
        plt.ylim(0, 1.15)  # Leave room for value labels
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a more visible legend
        plt.legend(
            title='Models',
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True, 
            shadow=True, 
            ncol=len(results)
        )
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_cross_validation_results(self, results):
        """Plot cross-validation results"""
        plt.figure(figsize=(10, 6))
        for name, scores in results.items():
            plt.errorbar(name, scores['cv_mean'], yerr=scores['cv_std'], 
                        fmt='o', capsize=5, label=name)
        
        plt.xlabel('Models')
        plt.ylabel('Cross-validation Score')
        plt.title('Cross-validation Results')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/cross_validation.png')
        plt.close()
    
    def plot_roc_curves(self, X, y):
        """Plot ROC curves for each class"""
        X_processed = self.preprocess_features(X, is_training=False)
        
        # Create a larger figure with better proportions
        plt.figure(figsize=(14, 10))
        
        # Limit the number of classes to plot to avoid overcrowding
        unique_classes = np.unique(y)
        if len(unique_classes) > 15:
            # If there are too many classes, select a representative sample
            # or the most important ones based on frequency
            class_counts = {c: np.sum(y == c) for c in unique_classes}
            top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_classes = [c for c, _ in top_classes]
        else:
            selected_classes = unique_classes
        
        # Plot ROC curves for selected classes
        for name, model in self.models.items():
            y_prob = model.predict_proba(X_processed)
            
            for i, class_name in enumerate(selected_classes):
                if i < len(np.unique(y)) and i < y_prob.shape[1]:
                    fpr, tpr, _ = roc_curve(y == class_name, y_prob[:, i])
                    plt.plot(fpr, tpr, label=f'{name} - {class_name}')
        
        # Add reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Improve layout and labels
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        
        # Move legend outside the plot to avoid overcrowding
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Ensure everything fits
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/roc_curves.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot confusion matrix with improved formatting"""
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create larger figure
        plt.figure(figsize=(14, 12))
        
        # Use a more readable color map and adjust annotation settings
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes, 
            yticklabels=classes,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        # Rotate tick labels for better readability if many classes
        if len(classes) > 10:
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=45, fontsize=8)
        else:
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
        
        # Add better labels and title
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        
        # Ensure everything fits
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig('plots/confusion_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        rf_model = self.models['random_forest']
        importance = rf_model.feature_importances_
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
    
    def plot_confidence_distribution(self, confidence, predictions, filename='confidence_distribution.png'):
        """Plot confidence distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(confidence, bins=50, alpha=0.7)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.tight_layout()
        plt.savefig(f'plots/{filename}')
        plt.close() 