import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.net(x)

def build_encoder(kind="swin_t"):
    if kind == "swin_t":
        from torchvision.models import swin_t, Swin_T_Weights
        m = swin_t(weights=Swin_T_Weights.DEFAULT)
        m.head = nn.Identity()
        return m, 768
    elif kind == "vit_b_16":
        from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
        m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        m.heads = nn.Identity()
        return m, 768
    else:
        raise ValueError("Unknown backbone")