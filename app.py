from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch

app = FastAPI()

# Load the pre-trained ResNet18 model from PyTorch Hub
model = models.resnet18(pretrained=True)
model.eval()

# Define the transformation to preprocess input images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.tolist()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        probabilities = predict_image(file.file)
        return JSONResponse(content={"class_probabilities": probabilities})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
