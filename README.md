# Automated Biometric Identification of Mugger Crocodiles

## Problem Statement
The Mugger Crocodile (*Crocodylus palustris*) is classified as **'Vulnerable'** due to habitat destruction and illegal wildlife trade. Identifying individual crocodiles is crucial for conservation efforts. However, deep learning-based models often misclassify unknown individuals, leading to **high false positive rates**. 

This project aims to develop a **lightweight machine learning-based system** for crocodile identification using **scute pattern analysis**, improving accuracy in recognizing both known and unknown individuals while reducing false positives.

---

## Let's Get Started

## Virtual Environment Setup Guide

### 📌 Cloning the Repository
Before setting up the virtual environment, clone the repository and navigate to the `Codes` folder:
```bash
git clone https://github.com/Dhruv0306/ML_2025_11_Group7.git
cd ML_2025_11_Group7/Codes
```

### 📌 Creating a Virtual Environment in Python
A virtual environment helps isolate project dependencies and avoid conflicts with global packages.

#### **1️⃣ Install Python (If Not Already Installed)**
Ensure you have Python installed. You can check by running:
```bash
python3 --version
```
If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and install it.

#### **2️⃣ Install `venv` (If Not Available)**
For Linux-based systems, install `venv` if it's not already installed:
```bash
sudo apt update
sudo apt install python3-venv  # For Debian/Ubuntu
sudo dnf install python3-venv  # For Fedora
```

#### **3️⃣ Create a Virtual Environment**
Navigate to your project folder and run:
```bash
python3 -m venv venv
```
This creates a folder named `venv` containing the isolated environment.

#### **4️⃣ Activate the Virtual Environment**
- **On Windows (Command Prompt):**
  ```bash
  venv\Scripts\activate
  ```
- **On Windows (PowerShell):**
  ```bash
  venv\Scripts\Activate.ps1
  ```
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

Once activated, your terminal prompt will change, showing `(venv)` before the directory path.

### 📌 Installing Libraries
#### **5️⃣ Install Libraries from `requirements.txt`**
If you want to install the required libraries, use:
```bash
pip install -r requirements.txt
```

### 📌 Deactivating and Managing the Virtual Environment
- **Deactivate the Virtual Environment:**
  ```bash
  deactivate
  ```
---

## Approach
To tackle the problem of crocodile identification, we first needed a reliable dataset of crocodile scute patterns. We started by collecting publicly available images and preprocessing them to enhance visibility of unique patterns. This involved resizing, normalizing, and applying image enhancement techniques using OpenCV.

Next, we focused on extracting meaningful features from these scute patterns. We experimented with multiple feature extraction techniques such as HOG, SIFT, ORB, and LBP to identify which provided the most distinct and consistent pattern representations.

Once features were extracted, we encountered a challenge: high-dimensional data. To address this, we applied dimensionality reduction techniques like PCA and t-SNE, ensuring that essential features were retained while improving computational efficiency.

With refined features, we trained various machine learning models including SVM, Random Forest, k-NN, and XGBoost to classify and re-identify crocodiles. Each model was tuned using cross-validation to optimize performance.

An additional challenge was identifying previously unseen crocodiles. We integrated an outlier detection mechanism using threshold-based confidence scoring to flag unknown individuals, reducing false positives.

For evaluation, we compared our approach with deep learning models like CNN and YOLO to analyze trade-offs in accuracy, speed, and efficiency. We assessed model performance using Precision, Recall, F1-score, and Confusion Matrix metrics.

### **1. Data Collection & Preprocessing**
- Gather images of Mugger Crocodiles from **publicly available datasets** or field research.
- **Preprocess images**: Resize, normalize, and enhance scute patterns using OpenCV.
- **Manually annotate** unique crocodile individuals for training/testing.

### **2. Feature Extraction**
- Extract handcrafted features from scute patterns using:
  - **HOG (Histogram of Oriented Gradients)**
  - **SIFT (Scale-Invariant Feature Transform)**
  - **ORB (Oriented FAST and Rotated BRIEF)**
  - **LBP (Local Binary Patterns)**

### **3. Dimensionality Reduction**
- Reduce feature complexity using:
  - **PCA (Principal Component Analysis)**
  - **t-SNE (t-distributed Stochastic Neighbor Embedding)**

### **4. Model Training**
- Train ML models for classification & individual re-identification:
  - **SVM (Support Vector Machine)**
  - **Random Forest**
  - **k-NN (K-Nearest Neighbors)**
  - **XGBoost**
- Optimize model performance using **cross-validation**.

### **5. Unknown Crocodile Detection**
- Implement an **outlier detection mechanism** to recognize previously unseen crocodiles.
- Use threshold-based confidence scoring to flag unknown individuals.

---

## Installation & Usage
### **Prerequisites**
- Python 3.8+
- OpenCV
- Scikit-learn
- NumPy, Pandas
- Matplotlib

### **Installation**
```bash
pip install -r requirements.txt
```

## Expected Outcome
- **A lightweight, non-deep-learning-based system** for crocodile identification.
- **Handcrafted feature-based models** compared against CNN-based approaches.
- **Optimized dimensionality reduction methods** preserving identity-specific information.

---

## Contributors
- **Dhruv Rakeshbhai Patel** ([Dhruv0306](https://github.com/Dhruv0306))
- **Mitul Ranpariya** ([MitulRanPariya](https://github.com/MitulRanpariya))
- **Parth Mevada** ([parthmevada2307](https://github.com/parthmevada2307))
- **Pratik Malviya** ([pratikmalviya24](https://github.com/pratikmalviya24))
- **Sameer Gediya** ([Sameer7188](https://github.com/Sameer7188))

---
