# Facial Emotion Recognition – Vintage Nights

## Overview
This project classifies 48×48 grayscale facial images into 7 emotions:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Dataset
- Total images: 28,709
- Image size: 48×48 (grayscale)
- Source: FER-style Kaggle dataset

## Model
- ResNet-18 (pretrained)
- Modified for small images
- Weighted Cross-Entropy for class imbalance
- 

## Training
- Epochs: 20
- Optimizer: Adam
- Hardware: NVIDIA RTX 4060

## Performance
- Validation Accuracy: ~60%

## TrainedModel

The trained ResNet-18 model (~42 MB) is available via Google Drive:

🔗 https://drive.google.com/drive/folders/1pVptxldqa7KNrcy3v8LQQGM64YeiH0G5?usp=sharing

The model can be loaded using:
```python
model.load_state_dict(torch.load("emotion_resnet18.pth"))
```
## Inference
```bash
python predict_test.py



