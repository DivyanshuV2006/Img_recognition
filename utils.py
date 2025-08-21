import pickle
import numpy as np
from PIL import Image
import torch

def save_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print("Error loading data:", str(e))
        return None

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.tensor(img_np).unsqueeze(0).float()  # Add batch dimension
    return img_tensor

def test_onnx_model(ort_session, image_tensor):
    # Convert the tensor to the required format for ONNX
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    outputs = ort_session.run(None, inputs)
    return outputs