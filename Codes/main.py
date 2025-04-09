import os
import cv2
import numpy as np
from pathlib import Path
from utils import create_directory, parse_voc_xml, crop_image, extract_croc_id_from_filename
from feature_extraction import FeatureExtractor
from models import CrocodileClassifier

class CrocodilePipeline:
    def __init__(self):
        """
        Initialize the crocodile identification pipeline
        """
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize classifier
        self.classifier = CrocodileClassifier()
        
        # Create output directories
        self.output_dirs = {
            'training': 'cropped/Training',
            'test_known': 'cropped/Test/Known',
            'test_unknown': 'cropped/Test/Unknown'
        }
        for dir_path in self.output_dirs.values():
            create_directory(dir_path)
    
    def process_training_data(self, training_dir):
        """
        Process training data: extract bounding boxes and features
        
        Args:
            training_dir (str): Path to training data directory
            
        Returns:
            tuple: (features, labels)
        """
        features = []
        labels = []
        
        # Process each crocodile folder
        for croc_dir in os.listdir(training_dir):
            croc_path = os.path.join(training_dir, croc_dir)
            if not os.path.isdir(croc_path):
                continue
            
            print(f"\nProcessing {croc_dir}...")
            
            # Process each image in the folder
            for img_file in os.listdir(croc_path):
                if not img_file.endswith('.jpg'):
                    continue
                
                # Get image and XML paths
                img_path = os.path.join(croc_path, img_file)
                xml_path = os.path.join(croc_path, img_file.replace('.jpg', '.xml'))
                
                # Parse bounding box
                bbox = parse_voc_xml(xml_path)
                
                # Crop image
                cropped_img = crop_image(img_path, bbox)
                
                # Save cropped image
                output_path = os.path.join(self.output_dirs['training'], croc_dir, img_file)
                create_directory(os.path.dirname(output_path))
                cv2.imwrite(output_path, cropped_img)
                
                # Extract features
                img_features = self.feature_extractor.extract_all_features(cropped_img)
                
                features.append(img_features)
                labels.append(croc_dir)
        
        return np.array(features), np.array(labels)
    
    def process_test_data(self, test_dir, is_known=True):
        """
        Process test data: crop images and extract features
        
        Args:
            test_dir (str): Path to test data directory
            is_known (bool): Whether the test data is for known crocodiles
            
        Returns:
            tuple: (features, labels) if is_known else (features,)
        """
        features = []
        labels = [] if is_known else None
        
        # Process each image
        for img_file in os.listdir(test_dir):
            if not img_file.endswith('.jpg'):
                continue
            
            print(f"\nProcessing {img_file}...")
            
            # Get image path
            img_path = os.path.join(test_dir, img_file)
            
            # Crop image (center crop for unknown)
            cropped_img = crop_image(img_path)
            
            # Save cropped image
            output_dir = self.output_dirs['test_known' if is_known else 'test_unknown']
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, cropped_img)
            
            # Extract features
            img_features = self.feature_extractor.extract_all_features(cropped_img)
            
            features.append(img_features)
            if is_known:
                labels.append(extract_croc_id_from_filename(img_file))
        
        if is_known:
            return np.array(features), np.array(labels)
        return np.array(features)
    
    def run_pipeline(self, training_dir, test_known_dir, test_unknown_dir):
        """
        Run the complete pipeline
        
        Args:
            training_dir (str): Path to training data directory
            test_known_dir (str): Path to known test data directory
            test_unknown_dir (str): Path to unknown test data directory
        """
        print("Processing training data...")
        X_train, y_train = self.process_training_data(training_dir)
        
        print("\nTraining and evaluating models...")
        results = self.classifier.train_and_evaluate(X_train, y_train)
        
        # Print results
        print("\nModel Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Plot confusion matrix for training data
        y_pred = self.classifier.predict(X_train)[0]
        self.classifier.plot_confusion_matrix(y_train, y_pred, np.unique(y_train))
        
        # Plot feature importance
        self.classifier.plot_feature_importance()
        
        # Process known test data
        print("\nProcessing known test data...")
        X_test_known, y_test_known = self.process_test_data(test_known_dir, is_known=True)
        
        # Make predictions for known test data
        y_pred_known, confidence_known = self.classifier.predict(X_test_known)
        
        # Print known test results
        print("\nKnown Test Results:")
        print(f"Accuracy: {np.mean(y_pred_known == y_test_known):.4f}")
        print(f"Average Confidence: {np.mean(confidence_known):.4f}")
        
        # Process unknown test data
        print("\nProcessing unknown test data...")
        X_test_unknown = self.process_test_data(test_unknown_dir, is_known=False)
        
        # Make predictions for unknown test data
        y_pred_unknown, confidence_unknown = self.classifier.predict(X_test_unknown)
        
        # Print unknown test results
        print("\nUnknown Test Results:")
        print(f"Number of Unknown Predictions: {np.sum(y_pred_unknown == 'Unknown')}")
        print(f"Average Confidence: {np.mean(confidence_unknown):.4f}")

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CrocodilePipeline()
    
    # Run pipeline
    pipeline.run_pipeline(
        training_dir="dataset/Training",
        test_known_dir="dataset/Test/Known",
        test_unknown_dir="dataset/Test/Unknown"
    ) 