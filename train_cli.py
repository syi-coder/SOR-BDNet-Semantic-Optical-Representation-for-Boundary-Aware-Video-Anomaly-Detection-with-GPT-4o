import argparse
from model import ContrastiveFolderDataset, Trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone", type=str, default="swin_t", choices=["swin_t","vit_b_16"])
    p.add_argument("--save_path", type=str, default="simclr_transformer.pth")
    args = p.parse_args()
    t = Trainer(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        dataset_cls=ContrastiveFolderDataset
    )
    t.fit(save_path=args.save_path)

if __name__ == "__main__":
    main()