# Facial Emotion Recognition

## Overview
This project focuses on classifying grayscale facial images captured into one of seven human emotions:
**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

The task is solved using a deep learning–based image classification approach.

---

## Dataset
- Total images: **28,709**
- Image size: **48 × 48 (grayscale)**
- Emotion classes: **7**
- Dataset style: FER-style dataset (same distribution as Kaggle FER datasets)

Class distribution is imbalanced, with the *Disgust* class having significantly fewer samples.

---

## Model Architecture
- **Backbone:** ResNet-18 (pretrained on ImageNet)
- Modified first convolution layer for small 48×48 images
- Removed initial max-pooling layer to preserve facial details
- Final fully connected layer outputs **7 emotion classes**

---

## Training Details
- Framework: **PyTorch**
- Optimizer: **Adam**
- Loss Function: **Weighted Cross-Entropy** (to handle class imbalance)
- Epochs: **20**
- Batch Size: **64**
- Hardware: **NVIDIA RTX 4060 (CUDA enabled)**

---

## Performance
- **Validation Accuracy:** ~**59.4%**

The trained ResNet-18 model achieved 59.40% accuracy on the labeled test dataset.

---

## Trained Model
As requested, the trained model is provided via Google Drive.

🔗 **Google Drive link (trained models):**  
https://drive.google.com/drive/folders/1pVptxIdqa7KNrcy3v8LQQGM64YeiH0G5?usp=sharing

The trained ResNet-18 model file (~42 MB) can be loaded using:

```python
model.load_state_dict(torch.load("emotion_resnet18.pth"))


