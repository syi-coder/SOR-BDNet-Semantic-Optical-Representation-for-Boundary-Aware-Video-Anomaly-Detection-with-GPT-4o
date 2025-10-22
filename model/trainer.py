import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .encoders import build_encoder
from .heads import ProjectionHead
from .loss import nt_xent_loss

class Trainer:
    def __init__(self, data_dir, epochs=5, batch_size=128, lr=3e-4, backbone="swin_t", dataset_cls=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert dataset_cls is not None
        self.dataset = dataset_cls(data_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        encoder, feat_dim = build_encoder(backbone)
        projector = ProjectionHead(input_dim=feat_dim)
        self.model = torch.nn.Sequential(encoder, projector).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def fit(self, save_path="simclr_transformer.pth"):
        self.model.train()
        for e in range(self.epochs):
            total = 0.0
            for xi, xj in tqdm(self.dataloader, desc=f"Epoch {e+1}/{self.epochs}"):
                xi, xj = xi.to(self.device), xj.to(self.device)
                zi = self.model(xi)
                zj = self.model(xj)
                loss = nt_xent_loss(zi, zj)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += float(loss.item())
            print(f"✅ Epoch {e+1}: Loss = {total/len(self.dataloader):.4f}")
        torch.save(self.model.state_dict(), save_path)
        print("✅ Model saved to", save_path)
