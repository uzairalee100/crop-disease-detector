"""
predictor.py
Combines CNN model prediction + Groq treatment advice
Main module used by Streamlit app
"""
import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import json, os
from treatment_advisor import get_advice

# Paths
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)
DEVICE     = torch.device('cpu')

# Load classes
with open(os.path.join(MODELS_DIR, 'classes.json')) as f:
    CLASSES = json.load(f)

# Build and load model
def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

MODEL = build_model(len(CLASSES))
MODEL.load_state_dict(
    torch.load(
        os.path.join(MODELS_DIR, 'best_model.pth'),
        map_location=DEVICE
    )
)
MODEL.eval()
print("Model loaded!")

# Image transform
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image: Image.Image, language: str = "English") -> dict:
    """
    Main prediction function.
    Takes PIL image → returns full result dict.
    """
    # Step 1 — Preprocess image
    img_tensor = TRANSFORM(image.convert('RGB')).unsqueeze(0)

    # Step 2 — CNN prediction
    with torch.no_grad():
        outputs    = MODEL(img_tensor)
        probs      = torch.softmax(outputs, dim=1)
        confidence = probs.max().item() * 100
        pred_idx   = outputs.argmax(1).item()

        if pred_idx < len(CLASSES):
            disease_class = CLASSES[pred_idx]
        else:
            disease_class = CLASSES[0]

    # Step 3 — Get Groq treatment advice
    advice_result = get_advice(disease_class, language)

    return {
        "disease_class": disease_class,
        "crop":          advice_result["crop"],
        "disease":       advice_result["disease"],
        "confidence":    round(confidence, 2),
        "severity":      advice_result["severity"],
        "advice":        advice_result["advice"],
        "language":      language,
        "is_healthy":    advice_result["disease"].lower() == "healthy"
    }

if __name__ == "__main__":
    # Quick test
    test_img = Image.open(
        r"D:\crop_disease_detector\data\plantvillage dataset\color\Apple___Black_rot\1aa28af5-8270-467c-b496-ff415f0b58df___JR_FrgE.S 3010.JPG"
    )
    result = predict(test_img, "English")
    print(f"Crop:       {result['crop']}")
    print(f"Disease:    {result['disease']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Severity:   {result['severity']}")
    print(f"\nAdvice preview:")
    print(result['advice'][:300] + "...")