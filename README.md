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
