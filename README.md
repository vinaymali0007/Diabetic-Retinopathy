# Diabetic Retinopathy Classification with Deep Learning  

This project focuses on developing a deep learning-based pipeline for binary classification of diabetic retinopathy stages. Leveraging state-of-the-art architectures like **VGG16** and **MobileNetV2**, combined with advanced techniques like attention mechanisms and data augmentation, the model achieves robust performance in detecting and categorizing diabetic retinopathy.  

## Overview  
The aim of this project is to improve diagnostic accuracy for diabetic retinopathy by utilizing transfer learning and modern deep learning strategies. By training on a Kaggle-sourced dataset, the models classify images into multiple stages of retinopathy, contributing to more effective healthcare solutions.  

## Key Features  
- **Transfer Learning:** Fine-tuned **VGG16** and **MobileNetV2** models for feature extraction.  
- **Attention Mechanisms:** Enhanced the focus on critical regions in images for improved classification.  
- **Data Augmentation:** Used **ImageDataGenerator** for advanced augmentation to address dataset imbalance.  
- **Regularization Techniques:** Incorporated **Batch Normalization**, **Dropout**, and **Gaussian Noise** for better generalization.  
- **Performance Metrics:** Evaluated using metrics such as accuracy, confusion matrix, and classification reports.  

## Dataset  
The dataset used for this project is publicly available on Kaggle and contains labeled images corresponding to different stages of diabetic retinopathy:  
- **Training Images:** Used for model training.  
- **Validation Images:** Used for hyperparameter tuning.  
- **Testing Images:** Used for final model evaluation.  

Ensure the dataset is downloaded and properly structured for seamless integration with the training pipeline.  

## Architectures  
### 1. **VGG16:**  
- Pre-trained backbone for feature extraction.  
- Enhanced with attention mechanisms and dense layers for classification.  

### 2. **MobileNetV2:**  
- Lightweight architecture for efficient feature extraction.  
- Fine-tuned for diabetic retinopathy classification with additional layers and regularization.  

## Training Pipeline  
- Efficient data handling for train, validation, and test splits.  
- Augmented pre-processing for image normalization and augmentation.  
- Multi-stage training to optimize model performance.  

## Results  
- **Accuracy:** Achieved significant improvement over baseline models.  
- **Evaluation Metrics:** Included confusion matrices and classification reports to analyze performance on imbalanced data.  


