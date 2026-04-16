# Natural Scene Classification Project

## Overview

This project performs image classification on natural scene images using deep learning techniques.

The goal is to classify images into six categories:

- buildings
- forest
- glacier
- mountain
- sea
- street

Two models are implemented:

1. A custom Convolutional Neural Network (CNN)
2. A pretrained ResNet18 model (transfer learning)

---

## Dataset

This project uses the **Intel Image Classification dataset**.

📥 Download from Kaggle:
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

After downloading and extracting the dataset, ensure the folder structure is as follows:

### Folder Structure

After downloading and extracting the dataset, it should be organized as follows:

data/
├── seg_train/
│ ├── buildings/
│ ├── forest/
│ ├── glacier/
│ ├── mountain/
│ ├── sea/
│ └── street/
├── seg_test/
│ ├── buildings/
│ ├── forest/
│ ├── glacier/
│ ├── mountain/
│ ├── sea/
│ └── street/

> Note: The dataset contains nested folders (`seg_train/seg_train` and `seg_test/seg_test`). Make sure your code points to the correct directory.

---

## Setup Instructions

### 1. Clone or download the project

```bash
git clone <your-repo-link>
cd <project-folder>
```

### 2. Create a virtual environment

python -m venv .venv

Activate the environment:

- Windows (PowerShell or CMD):
  .venv\Scripts\activate

- Git Bash:
  source .venv/Scripts/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Set up Jupyter kernel

python -m ipykernel install --user --name=intel_project --display-name "Python (.venv Intel Project)"

### 5. Run the notebook

Open the notebook file and select the correct kernel:
Python (.venv Intel Project)


Author
Cynthia Urrutia