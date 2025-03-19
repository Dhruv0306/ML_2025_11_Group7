import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = "./dataset/"  # Update this to the actual dataset location

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' does not exist!")
else:
    print(f"Dataset found at: {DATASET_PATH}")

# Function to load images without resizing & keep original RGB format
def load_images(dataset_path):
    images = []
    labels = []
    
    for croc_id in os.listdir(dataset_path):
        croc_folder = os.path.join(dataset_path, croc_id)
        print("Loading images from:", croc_folder)
        if os.path.isdir(croc_folder):
            for img_file in os.listdir(croc_folder):
                img_path = os.path.join(croc_folder, img_file)
                img = cv2.imread(img_path)  # Load in original color (BGR format)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                
                if img is not None:
                    img = img / 255.0  # Normalize pixel values (0 to 1)
                    images.append(img)
                    labels.append(croc_id)
    
    return np.array(images), np.array(labels)

# Load dataset (with original size and color)
images, labels = load_images(DATASET_PATH)
print(f"Images: {images.shape}, Labels: {labels.shape}")
# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Display sample images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f"ID: {y_train[i]}")
    plt.axis("off")
plt.show()

# Print dataset information
print("Dataset Loaded Successfully!")
print(f"Total Images: {len(images)}, Train Set: {len(X_train)}, Test Set: {len(X_test)}")
print(f"Image Dimensions After Processing: {X_train[0].shape}")  # Should be (3840, 2160, 3)