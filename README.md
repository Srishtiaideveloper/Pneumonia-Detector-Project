# Deep Learning Driven Pneumonia Classification on Chest X-Rays

## Project Overview

Developed a deep learning-based pneumonia detection system using ResNet50 with transfer learning in PyTorch, achieving 97% classification accuracy on chest X-ray images.
Applied data augmentation and K-Fold cross-validation to improve generalization.
Performed hyperparameter tuning (learning rate, batch size, optimizer) for performance optimization.
Integrated Grad-CAM to visualize decision-critical regions on X-rays, enhancing model transparency and interpretability.

## Dataset Overview

- **Normal images count in training set:** 6399
- **Pneumonia images count in training set:** 6343
- **Total Count of images:** 12742

## 🧪 Model Details

- Framework: **PyTorch**
- Architecture: **ResNet50** (pre-trained on ImageNet)
- Optimizations:  
  - Data augmentation (horizontal flip, rotation, scaling)  
  - K-Fold Cross Validation (K=5)  
  - Hyperparameter tuning (learning rate, batch size, optimizer)

## Project Structure

```
Chest_Xray_Pneumonia_Detector/
│── data/
│   ├── pneumonia/
│   ├── normal/
│── train/
│   ├── model_training_code.ipynb
│   ├── saving_model/
│       ├── pneumonia_model.keras
│── results/
│       ├── sample1.png
│       ├── sample3.png
│       ├──.....
│── deployment/
    ├── app.py
    ├── requirements.txt
    ├── index.html
```

## Installation & Setup

```bash
pip install -r requirements.txt
```

## Model Training (Google Colab)

1. Install dependencies
   ```bash
   !pip install -r requirements.txt
   ```
2. Prepare dataset
   ```
   Chest_Xray_Pneumonia_Detector/
       pneumonia /
           img1.jpg
           img2.jpg
       normal/
           img1.jpg
           img2.jpg
   ```
3. Train the model using `model_training_code.ipynb`

## Results & Interpretability
- Achieved **97% accuracy**
- Used **Grad-CAM** to visualize model focus areas for interpretability.

## 📂 Tools & Libraries

- **PyTorch**, **NumPy**, **OpenCV**
- **scikit-learn**, **Matplotlib**, **Seaborn**
- **Grad-CAM**, **Flask**, **AWS SageMaker**

---

## 🧑‍💻 Developed By

**Srishti**  
[GitHub](https://github.com/Srishtiaideveloper)
