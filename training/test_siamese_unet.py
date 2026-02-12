# ====================== CONFIG (ALL CAPS) ======================
DATA_ROOT = "processed_dataset"      # Your folder with A/ and B/
MODEL_PATH = "siamese_unet.pt"       # Trained model
OUTPUT_VIS_DIR = "test_results"      # Folder to save outputs

PATCH_SIZE = 256
BATCH_SIZE = 1
SEED = 42
# ===============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------
# COLLECT T1/T2 PAIRS (SAME AS TRAINING)
# ---------------------------------------------------------
def collect_pairs(data_root):
    A_dir = os.path.join(data_root, "A")
    B_dir = os.path.join(data_root, "B")

    a_files = sorted([f for f in os.listdir(A_dir) if f.endswith("_T1.npy")])
    b_files = sorted([f for f in os.listdir(B_dir) if f.endswith("_T2.npy")])

    A_map = {f.replace("_T1.npy", ""): os.path.join(A_dir, f) for f in a_files}
    B_map = {f.replace("_T2.npy", ""): os.path.join(B_dir, f) for f in b_files}

    common_keys = sorted(set(A_map.keys()).intersection(set(B_map.keys())))

    pairs = [(A_map[k], B_map[k], k) for k in common_keys]  # keep location ID

    print(f"[INFO] Total matched patch pairs: {len(pairs)}")
    return pairs

# ---------------------------------------------------------
# DATASET FOR TESTING
# ---------------------------------------------------------
class SiameseTestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def fix_shape(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        C, H, W = img.shape
        if H == 257 and W == 257:
            img = img[:, :256, :256]

        return img.astype(np.float32)

    def __getitem__(self, idx):
        t1_path, t2_path, loc_id = self.pairs[idx]

        img1 = np.load(t1_path)
        img2 = np.load(t2_path)

        img1 = self.fix_shape(img1)
        img2 = self.fix_shape(img2)

        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)

        # Dummy ground truth (since you don’t have real masks yet)
        y = torch.zeros((1, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)

        return img1, img2, y, loc_id

# ---------------------------------------------------------
# SAME MODEL (COPY FROM TRAINING)
# ---------------------------------------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)
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

        out = self.out(d1)
        return self.sigmoid(out)

# ---------------------------------------------------------
# VISUALIZE RESULT
# ---------------------------------------------------------
def visualize_result(t1, t2, pred, loc_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    t1_img = t1.squeeze().cpu().numpy()
    t2_img = t2.squeeze().cpu().numpy()
    pred_img = pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(t1_img, cmap="gray")
    axes[0].set_title("T1")

    axes[1].imshow(t2_img, cmap="gray")
    axes[1].set_title("T2")

    axes[2].imshow(pred_img, cmap="hot")
    axes[2].set_title("Predicted Change")

    for ax in axes:
        ax.axis("off")

    save_path = os.path.join(out_dir, f"{loc_id}_result.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved visualization: {save_path}")

# ---------------------------------------------------------
# TEST FUNCTION
# ---------------------------------------------------------
def test_model(model, test_loader):
    model.eval()
    criterion = nn.BCELoss()

    total_loss = 0.0

    with torch.no_grad():
        for img1, img2, y, loc_id in test_loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(img1, img2)
            loss = criterion(preds, y)
            total_loss += loss.item()

            # Save visualization
            visualize_result(img1[0], img2[0], preds[0], loc_id[0], OUTPUT_VIS_DIR)

    avg_loss = total_loss / len(test_loader)
    print(f"\n[RESULT] Test Loss: {avg_loss:.4f}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print("[INFO] Loading test data...")
    pairs = collect_pairs(DATA_ROOT)

    if len(pairs) == 0:
        raise ValueError("No matching T1/T2 pairs found.")

    # Use last 5 pairs as test (same split logic)
    test_pairs = pairs[-5:]

    test_dataset = SiameseTestDataset(test_pairs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Loading trained model...")
    model = SiameseUNet(in_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    print("[INFO] Running testing...")
    test_model(model, test_loader)

    print("[DONE] Testing completed.")
