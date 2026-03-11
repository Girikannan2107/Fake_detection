import torch
from transformers import ViTForImageClassification

def load_deepfake_model():

    # Load base Vision Transformer
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )

    # Change classifier to 2 classes (Real / Fake)
    model.classifier = torch.nn.Linear(768, 2)

    # Load trained deepfake weights
    model.load_state_dict(
        torch.load("models/vit_deepfake_model.pth", map_location=torch.device("cpu"))
    )

    model.eval()

    return model