import argparse
import os
import onnxruntime as ort
from data_processing import BuildDataset, createLabels
from train import train
from utils import save_data, preprocess_image, test_onnx_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--build", action='store_true', help="Build data from scratch")
    parser.add_argument("--train", action='store_true', help="Train the model")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    data_path = r"C:\Users\Test01\Desktop\CS3-Midterm\CNN\animals"
    file_path = "utils/training_data.pkl"
    
    if args.build:
        if os.path.exists(file_path):
            print("Pickle file already exists. Skipping dataset building.")
        else:
            LABELS = createLabels(data_path)
            Builder = BuildDataset(data_path, LABELS)
            save_data(Builder.data, file_path)

    if args.train:
        LABELS = createLabels(data_path)
        train(data_path, LABELS, num_epochs=args.num_epochs, lr=args.lr)

    if args.img:
        onnx_model_path = r"utils\model.onnx"
        test_image_path = args.img

        ort_session = ort.InferenceSession(onnx_model_path)
        image_tensor = preprocess_image(test_image_path)

        output = test_onnx_model(ort_session, image_tensor)
        predictions = output[0][0]
        predicted_index = predictions.argmax()
        predicted_label = list(LABELS.keys())[list(LABELS.values()).index(predicted_index)]
        confidence = predictions[predicted_index]

        print(f"Predicted Animal: {predicted_label}, (Confidence: {confidence:.2f})")
if __name__ == "__main__":
    main()