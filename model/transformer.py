from torchvision import transforms as T
import torch

class RandomChannelPermutation:
    def __call__(self, x):
        idx = torch.randperm(x.size(0))
        return x.index_select(0, idx)

class GaussianNoise:
    def __init__(self, sigma=(0.0, 0.05)):
        self.sigma = sigma
    def __call__(self, x):
        s = torch.empty(1).uniform_(*self.sigma).item()
        if s <= 1e-6:
            return x
        return (x + torch.randn_like(x) * s).clamp(0.0, 1.0)

def build_contrastive_transforms():
    weak_geom = T.RandomApply([
        T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6)
    ], p=0.25)
    strong_geom = T.RandomApply([
        T.RandomPerspective(distortion_scale=0.25, p=1.0)
    ], p=0.25)
    color_block = T.RandomApply([
        T.ColorJitter(0.5, 0.5, 0.5, 0.2),
        T.RandomAdjustSharpness(1.75, p=1.0),
        T.RandomAutocontrast(p=1.0),
        T.RandomEqualize(p=1.0)
    ], p=0.6)
    bit_block = T.RandomApply([
        T.RandomPosterize(bits=4),
        T.RandomSolarize(threshold=192/255.0)
    ], p=0.3)
    blur_block = T.RandomApply([
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
    ], p=0.5)
    branch_a = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        T.RandomHorizontalFlip(p=0.5),
        weak_geom,
        color_block,
        T.RandomGrayscale(p=0.2),
        blur_block,
        T.ToTensor(),
        T.RandomApply([GaussianNoise(sigma=(0.0, 0.05))], p=0.5),
        T.Normalize((0.5,), (0.5,)),
        T.RandomApply([RandomChannelPermutation()], p=0.15),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random")
    ])
    branch_b = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        strong_geom,
        bit_block,
        T.RandomGrayscale(p=0.1),
        blur_block,
        T.ToTensor(),
        T.RandomApply([GaussianNoise(sigma=(0.0, 0.08))], p=0.35),
        T.Normalize((0.5,), (0.5,)),
        T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")
    ])
    chooser = T.RandomChoice([branch_a, branch_b], p=[0.6, 0.4])
    return chooser
