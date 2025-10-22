import os
import random
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_contrastive_transforms

class ContrastiveFolderDataset(Dataset):
    def __init__(self, root_dir, exts=(".png",)):
        self.image_paths = []
        for subdir in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, subdir)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if any(fname.lower().endswith(e) for e in exts):
                        self.image_paths.append(os.path.join(folder_path, fname))
        random.shuffle(self.image_paths)
        self.transform = build_contrastive_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj
