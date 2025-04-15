# Crocodile Identification Pipeline: Component Explanation

## 1. Initialization

- **Feature Extractor**: Initializes the feature extraction components that will process crocodile images.
  - **SIFT (Scale-Invariant Feature Transform)**: Detects and describes local features in images that are invariant to scaling and rotation.
  - **HOG (Histogram of Oriented Gradients)**: Captures edge or gradient structure that is characteristic of local shape.
  - **LBP (Local Binary Patterns)**: Captures texture patterns and is robust to monotonic illumination changes.
  - **ORB (Oriented FAST and Rotated BRIEF)**: A fast and efficient alternative to SIFT and SURF.

- **Classifier**: Sets up the machine learning models for crocodile identification.
  - **Random Forest**: Ensemble learning method that constructs multiple decision trees and outputs the class that is the mode of the classes of the individual trees.
  - **KNN (K-Nearest Neighbors)**: Classification by a plurality vote of the k nearest neighbors of each point.

- **Output Directories**: Creates folders to store intermediate and final results.
  - **cropped/Training**: Stores cropped training images.
  - **cropped/Test/Known**: Stores cropped test images with known labels.
  - **cropped/Test/Unknown**: Stores cropped test images with unknown labels.
  - **output/**: Stores the final annotated images with predictions.

## 2. Training Data Processing

- **XML Parsing**: Extracts bounding box coordinates from annotation files.
  - Each training image has a corresponding XML file with `<object>` and `<bndbox>` tags.
  - Extracts `xmin`, `ymin`, `xmax`, `ymax` coordinates for accurate cropping.

- **Image Cropping**: Crops the original images using the bounding box coordinates.
  - Focuses on the crocodile region, eliminating background noise.
  - Resizes all cropped images to a standard size (224x224 pixels).

- **Feature Extraction**: Extracts features from cropped images.
  - Applies all feature extractors (SIFT, HOG, LBP, ORB) to each image.
  - Combines these features into a single feature vector per image.
  - Caches extracted features to avoid redundant computation.

## 3. Model Training and Evaluation

- **Feature Preprocessing**:
  - **Scaling**: Standardizes features to have zero mean and unit variance.
  - **PCA**: Reduces dimensionality while preserving most of the variance (1500 components).

- **Random Forest Training**:
  - Uses 100 trees with configurable depth and split criteria.
  - Parallel processing for faster training.
  - Cross-validation to assess generalization performance.

- **KNN Training**:
  - Uses 5 nearest neighbors for classification.
  - Parallel processing for faster prediction.
  - Cross-validation to tune parameters.

- **Metric Calculation**:
  - **Accuracy**: Overall proportion of correct predictions.
  - **Precision**: Ability to avoid false positives.
  - **Recall**: Ability to find all positive samples.
  - **F1 Score**: Harmonic mean of precision and recall.
  - **ROC AUC**: Area under the Receiver Operating Characteristic curve.

## 4. Visualization Generation

- **Model Comparison**: Bar chart comparing performance metrics across models.
- **Cross-validation Results**: Visualization of model stability across different data splits.
- **ROC Curves**: Plots showing trade-off between true positive rate and false positive rate.
- **Confusion Matrix**: Tabular visualization of prediction accuracy for each class.
- **Feature Importance**: Bar chart showing which features contribute most to the classification.

## 5. Known Test Data Processing

- **Crocodile Detection**: Uses trained detector or XML annotations to locate crocodiles in test images.
- **Image Cropping**: Crops images to focus on detected crocodiles.
- **Feature Extraction**: Extracts the same feature set as used in training.

## 6. Known Test Prediction

- **Feature Preprocessing**: Applies the same preprocessing steps as during training.
- **Model Prediction**: Each model makes predictions on the test data.
- **Ensemble Voting**: Combines predictions from all models, weighted by confidence.
- **Metric Calculation**: Computes accuracy and confidence metrics on known test data.
- **Confidence Distribution**: Visualization of prediction confidence levels.

## 7. Unknown Test Data Processing

- **Crocodile Detection**: Uses trained detector to locate crocodiles in unknown test images.
- **Image Cropping**: Crops images to focus on detected crocodiles or uses center crop if detection fails.
- **Feature Extraction**: Extracts the same feature set as used in training.

## 8. Unknown Test Prediction

- **Feature Preprocessing**: Applies the same preprocessing steps as during training.
- **Model Prediction**: Each model makes predictions on the unknown test data.
- **Label Assignment**: Assigns "Unknown" label to these predictions.
- **Confidence Calculation**: Computes confidence level for each prediction.
- **Confidence Distribution**: Visualization of prediction confidence levels.

## 9. Save Annotated Images

- **Output Directory Creation**: Creates directories for storing annotated images.
- **Known Image Annotation**: Draws prediction text and confidence on known test images.
- **Unknown Image Annotation**: Draws "Unknown" label and confidence on unknown test images.
- **Image Saving**: Saves all annotated images to their respective output directories.

## Performance Considerations

- **Memory Efficiency**: Uses batch processing and IncrementalPCA to handle large datasets.
- **Execution Speed**: Implements parallel processing where applicable.
- **Caching**: Stores intermediate results to avoid redundant computation.
- **Error Handling**: Gracefully handles missing files, failed detections, etc.

## Outputs

1. **Model Performance Metrics**: Quantitative assessment of model performance.
2. **Visualizations**: Plots and charts for understanding model behavior.
3. **Annotated Test Images**: Original images with prediction overlay.
4. **Logs**: Detailed logs of the pipeline execution for debugging and monitoring. 