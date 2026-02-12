import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
DATA_DIR = "processed_dataset"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 150
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================

print(f"[INFO] Using device: {DEVICE}")

# ========================= DATASET =========================
class ChangeDataset(Dataset):
    def __init__(self, a_dir, b_dir, mask_dir, file_list):
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.mask_dir = mask_dir
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img1 = np.load(os.path.join(self.a_dir, fname + "_T1.npy"))
        img2 = np.load(os.path.join(self.b_dir, fname + "_T2.npy"))
        mask = np.load(os.path.join(self.mask_dir, fname + "_mask.npy"))

        # Ensure correct shape: (1, H, W)
        img1 = np.expand_dims(img1, axis=0).copy()
        img2 = np.expand_dims(img2, axis=0).copy()
        mask = np.expand_dims(mask, axis=0).copy()

        return (
            torch.tensor(img1, dtype=torch.float32),
            torch.tensor(img2, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


def load_file_pairs():
    a_files = sorted([os.path.basename(f).replace("_T1.npy", "")
                      for f in glob.glob(os.path.join(DATA_DIR, "A", "*.npy"))])

    b_files = sorted([os.path.basename(f).replace("_T2.npy", "")
                      for f in glob.glob(os.path.join(DATA_DIR, "B", "*.npy"))])

    mask_files = sorted([os.path.basename(f).replace("_mask.npy", "")
                         for f in glob.glob(os.path.join(DATA_DIR, "MASK", "*.npy"))])

    common = sorted(list(set(a_files) & set(b_files) & set(mask_files)))

    print(f"[INFO] Total matched pairs: {len(common)}")
    return common


# ========================= MODEL =========================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class SiameseResUNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), ResBlock(32))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), ResBlock(64))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), ResBlock(128))
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), ResBlock(256))
        self.pool = nn.MaxPool2d(2)

        # Decoder (CHANNEL-CORRECT)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResBlock(256)   # 128 + 128

        self.up2 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec2 = ResBlock(128)   # 64 + 64

        self.up1 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.dec1 = ResBlock(64)    # 32 + 32

        self.out = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        return c1, c2, c3, c4

    def forward(self, x1, x2):
        c1_1, c2_1, c3_1, c4_1 = self.encode(x1)
        c1_2, c2_2, c3_2, c4_2 = self.encode(x2)

        diff = c4_1 - c4_2

        u3 = self.up3(diff)
        u3 = torch.cat([u3, c3_1], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2_1], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1_1], dim=1)
        d1 = self.dec1(u1)

        return self.sigmoid(self.out(d1))


# ========================= LOSS =========================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        bce = self.bce(pred, target)
        return bce + (1 - dice)


# ========================= TRAIN =========================
def train(model, train_loader, val_loader):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for a, b, y in train_loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            p = model(a, b)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for a, b, y in val_loader:
                a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
                p = model(a, b)
                val_loss += criterion(p, y).item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[EPOCH {epoch}/{EPOCHS}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_siamese_resunet.pt")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("[INFO] Saved loss_curve.png")


# ========================= MAIN =========================
if __name__ == "__main__":
    files = load_file_pairs()

    # Split
    train_files = files[:21]
    val_files = files[21:25]
    test_files = files[25:]

    print(f"[INFO] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = ChangeDataset(
        os.path.join(DATA_DIR, "A"),
        os.path.join(DATA_DIR, "B"),
        os.path.join(DATA_DIR, "MASK"),
        train_files
    )

    val_ds = ChangeDataset(
        os.path.join(DATA_DIR, "A"),
        os.path.join(DATA_DIR, "B"),
        os.path.join(DATA_DIR, "MASK"),
        val_files
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Building Siamese ResUNet...")
    model = SiameseResUNet(in_ch=1).to(DEVICE)

    # Quick shape sanity check
    x = torch.randn(2, 1, 256, 256).to(DEVICE)
    y = model(x, x)
    print("Sanity output shape:", y.shape)

    train(model, train_loader, val_loader)

    print("[DONE] Training completed.")
