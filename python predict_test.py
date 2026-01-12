import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import pandas as pd
import os


def main():

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # PATHS (CHANGE IF NEEDED)
    TEST_DIR = r"D:\Machine Learning\Codefest\emotion_dataset\test"
    MODEL_PATH = "emotion_resnet18.pth"
    OUTPUT_CSV = "submission.csv"

    # CLASS LABELS (ORDER MATTERS)
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # TRANSFORMS
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    # LOAD TEST DATASET
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0   # IMPORTANT for Windows
    )

    print("Test images:", len(test_dataset))

    
    # LOAD MODEL
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 7)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    
    # INFERENCE
    results = []
    idx = 0

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for p in preds:
                img_path, _ = test_dataset.samples[idx]
                img_name = os.path.basename(img_path)
                results.append([img_name, labels[p]])
                idx += 1


    # SAVE CSV
    df = pd.DataFrame(results, columns=["image_name", "emotion"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Submission file saved as {OUTPUT_CSV}")
    print("Total predictions:", len(df))



if __name__ == "__main__":
    main()
