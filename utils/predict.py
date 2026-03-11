import torch
from PIL import Image
from models.load_model import load_deepfake_model
from utils.preprocess import transform

model = load_deepfake_model()

def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = torch.softmax(output.logits, dim=1)

        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()

    if fake_prob > real_prob:
        return "Fake", fake_prob
    else:
        return "Real", real_prob