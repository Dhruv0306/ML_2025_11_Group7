import os
import cv2
import numpy as np
from pathlib import Path
from utils import create_directory, parse_voc_xml, crop_image, extract_croc_id_from_filename, draw_prediction_on_image
from feature_extraction import FeatureExtractor
from models import CrocodileClassifier
from tqdm import tqdm

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
        total_images = 0
        processed_folders = 0
        
        print("\n=== Starting Training Data Processing ===")
        print(f"Training directory: {training_dir}")
        
        # Process each crocodile folder
        for croc_dir in os.listdir(training_dir):
            croc_path = os.path.join(training_dir, croc_dir)
            if not os.path.isdir(croc_path):
                continue
            
            processed_folders += 1
            folder_images = 0
            
            # Check if this folder has already been processed
            cropped_dir = os.path.join(self.output_dirs['training'], croc_dir)
            if os.path.exists(cropped_dir) and os.path.isdir(cropped_dir):
                print(f"\n[Folder {processed_folders}] Loading features from processed folder: {croc_dir}")
                # Load features from processed images
                for img_file in os.listdir(cropped_dir):
                    if not img_file.endswith('.jpg'):
                        continue
                    
                    folder_images += 1
                    total_images += 1
                    
                    # Load cropped image
                    img_path = os.path.join(cropped_dir, img_file)
                    cropped_img = cv2.imread(img_path)
                    
                    if cropped_img is None:
                        print(f"  Warning: Could not read image {img_path}")
                        continue
                    
                    # Extract features
                    try:
                        img_features = self.feature_extractor.extract_all_features(cropped_img)
                        features.append(img_features)
                        labels.append(croc_dir)
                    except Exception as e:
                        print(f"  Error extracting features from {img_file}: {str(e)}")
                        continue
                
                print(f"  Loaded {folder_images} images from {croc_dir}")
                continue
            
            print(f"\n[Folder {processed_folders}] Processing new folder: {croc_dir}...")
            
            # Process each image in the folder
            for img_file in os.listdir(croc_path):
                if not img_file.endswith('.jpg'):
                    continue
                
                folder_images += 1
                total_images += 1
                
                # Get image and XML paths
                img_path = os.path.join(croc_path, img_file)
                xml_path = os.path.join(croc_path, img_file.replace('.jpg', '.xml'))
                
                try:
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
                except Exception as e:
                    print(f"  Error processing {img_file}: {str(e)}")
                    continue
            
            print(f"  Processed {folder_images} images from {croc_dir}")
        
        if len(features) == 0:
            raise ValueError("No features extracted! Check if the dataset directories are correct.")
        
        print("\n=== Training Data Processing Summary ===")
        print(f"Total folders processed: {processed_folders}")
        print(f"Total images processed: {total_images}")
        print(f"Total features extracted: {len(features)}")
        print(f"Feature dimension: {len(features[0])}")
        print("=====================================\n")
            
        return np.array(features), np.array(labels)
    
    def process_test_data(self, test_dir, is_known=True):
        """
        Process test data using trained detection model and return image paths along with features/labels.
        
        Args:
            test_dir (str): Path to test data directory
            is_known (bool): Whether the test data is for known crocodiles
            
        Returns:
            tuple: (features, labels, image_paths) if is_known else (features, image_paths)
        """
        features = []
        image_paths = [] # Store original image paths
        labels = [] if is_known else None
        total_images = 0
        
        print(f"\n=== Processing {'Known' if is_known else 'Unknown'} Test Data ===")
        print(f"Test directory: {test_dir}")
        
        # Process each image
        for img_file in os.listdir(test_dir):
            if not img_file.endswith('.jpg'):
                continue
            
            total_images += 1
            print(f"\nProcessing image {total_images}: {img_file}")
            img_path = os.path.join(test_dir, img_file)
            image_paths.append(img_path) # Save original path
            
            try:
                # Detect and crop using trained detector
                cropped_img = self.detector.detect_and_crop(img_path)
                
                if cropped_img is None:
                    print(f"  Warning: No detection for {img_file}, using center crop")
                    cropped_img = crop_image(img_path, output_size=(224, 224))
                
                # Save cropped image (optional, maybe remove if not needed)
                # output_dir = self.output_dirs['test_known' if is_known else 'test_unknown']
                # output_path = os.path.join(output_dir, img_file)
                # cv2.imwrite(output_path, cropped_img)
                
                # Extract features
                img_features = self.feature_extractor.extract_all_features(cropped_img)
                
                features.append(img_features)
                if is_known:
                    labels.append(extract_croc_id_from_filename(img_file))
                
                print(f"  Successfully processed {img_file}")
                
            except Exception as e:
                print(f"  Error processing {img_file}: {str(e)}")
                # Append None or handle error appropriately for features/labels if needed
                # features.append(None) # Example placeholder
                continue
        
        print("\n=== Test Data Processing Summary ===")
        print(f"Total images processed: {total_images}")
        print(f"Total features extracted: {len(features)}")
        if len(features) > 0 and features[0] is not None:
             print(f"Feature dimension: {len(features[0])}")
        print("=====================================\n")
        
        if is_known:
            return np.array(features), np.array(labels), image_paths
        return np.array(features), image_paths
    
    def run_pipeline(self, training_dir, test_known_dir, test_unknown_dir):
        """
        Run the complete pipeline, including saving annotated test images.
        """
        print("\n=== Starting Crocodile Identification Pipeline ===")
        print(f"Training directory: {training_dir}")
        print(f"Known test directory: {test_known_dir}")
        print(f"Unknown test directory: {test_unknown_dir}")
        print("=============================================\n")

        # Stage 1: Train detection model
        # print("Training detection model...")
        # self.train_detector(training_dir) # Assuming detector is pre-trained or loaded

        # Stage 2: Process training data and train classifier
        print("\nProcessing training data...")
        X_train, y_train = self.process_training_data(training_dir)

        print("\nTraining and evaluating classifier...")
        results = self.classifier.train_and_evaluate(X_train, y_train)

        # ... (Print results and generate visualizations) ...
        print("\n=== Model Evaluation Results ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        print("==============================\n")

        print("\nGenerating visualizations...")
        # ... (Plotting code) ...
        self.classifier.plot_model_comparison(results)
        print("Saved model comparison plot")
        self.classifier.plot_cross_validation_results(results)
        print("Saved cross-validation results plot")
        self.classifier.plot_roc_curves(X_train, y_train)
        print("Saved ROC curves plot")
        # Need to handle cases where predict returns multiple model predictions
        if X_train.size > 0:
            y_pred_train, _ = self.classifier.predict(X_train)
            self.classifier.plot_confusion_matrix(y_train, y_pred_train, np.unique(y_train))
            print("Saved confusion matrix plot")
        self.classifier.plot_feature_importance()
        print("Saved feature importance plot")

        # --- Process Known Test Data and Save Annotated Images ---
        print("\nProcessing known test data...")
        X_test_known, y_test_known, known_image_paths = self.process_test_data(test_known_dir, is_known=True)

        print("\nMaking predictions for known test data...")
        y_pred_known, confidence_known = self.classifier.predict(X_test_known)

        # Save annotated known test images
        output_known_dir = 'output/Known'
        create_directory(output_known_dir)
        print(f"\nSaving annotated known test images to: {output_known_dir}")
        for i, img_path in enumerate(tqdm(known_image_paths, desc="Annotating Known Images")):
            base_filename = os.path.basename(img_path)
            output_image_path = os.path.join(output_known_dir, base_filename)
            draw_prediction_on_image(img_path, y_pred_known[i], confidence_known[i], output_image_path)

        # ... (Plot confidence distribution and print known test results) ...
        if confidence_known.size > 0:
             self.classifier.plot_confidence_distribution(confidence_known, y_pred_known)
             print("Saved confidence distribution plot for known test data")
        print("\n=== Known Test Results ===")
        if y_test_known.size > 0 and y_pred_known.size > 0:
             print(f"Accuracy: {np.mean(y_pred_known == y_test_known):.4f}")
        if confidence_known.size > 0:
            print(f"Average Confidence: {np.mean(confidence_known):.4f}")
        print("========================\n")

        # --- Process Unknown Test Data and Save Annotated Images ---
        print("\nProcessing unknown test data...")
        X_test_unknown, unknown_image_paths = self.process_test_data(test_unknown_dir, is_known=False)

        print("\nMaking predictions for unknown test data...")
        y_pred_unknown, confidence_unknown = self.classifier.predict(X_test_unknown)

        # Save annotated unknown test images
        output_unknown_dir = 'output/Unknown'
        create_directory(output_unknown_dir)
        print(f"\nSaving annotated unknown test images to: {output_unknown_dir}")
        for i, img_path in enumerate(tqdm(unknown_image_paths, desc="Annotating Unknown Images")):
            base_filename = os.path.basename(img_path)
            output_image_path = os.path.join(output_unknown_dir, base_filename)
            # Use 'Unknown' as the prediction text for this folder
            draw_prediction_on_image(img_path, "Unknown", confidence_unknown[i], output_image_path)

        # ... (Plot confidence distribution and print unknown test results) ...
        if confidence_unknown.size > 0:
            self.classifier.plot_confidence_distribution(confidence_unknown, y_pred_unknown,
                                                      filename='confidence_distribution_unknown.png')
            print("Saved confidence distribution plot for unknown test data")
        print("\n=== Unknown Test Results ===")
        # Count how many were predicted as 'Unknown' if your model supports it
        # If predict only gives croc IDs, this part needs adjustment
        # print(f"Number of Unknown Predictions: {np.sum(y_pred_unknown == 'Unknown')}") 
        if confidence_unknown.size > 0:
            print(f"Average Confidence: {np.mean(confidence_unknown):.4f}")
        print("==========================\n")

        print("\nAnnotated test images saved to the 'output' directory.")
        print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CrocodilePipeline()
    
    # Run pipeline
    pipeline.run_pipeline(
        training_dir="dataset/Training",
        test_known_dir="dataset/Test/Known",
        test_unknown_dir="dataset/Test/Unknown"
    ) 