import os
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, data_path, labels, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.data = self._build_data()

    def _build_data(self):
        data = []
        for label in tqdm(self.labels, desc="Building Dataset", delay=0.1):
            label_idx = self.labels[label]
            for img_name in os.listdir(os.path.join(self.data_path, label)):
                img_path = os.path.join(self.data_path, label, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))

                    if self.transform:
                        img = Image.fromarray(img)
                        img = self.transform(img)

                    data.append([img, label_idx])
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
