"""Reusable hybrid EfficientNet training code for the APTOS notebook."""

import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ben_graham_preprocess(img, target_size=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    if mask.any():
        rows, cols = np.where(mask)
        img = img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(img, (0, 0), target_size / 22.4)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    circle = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(circle, (target_size // 2, target_size // 2), int(target_size * 0.46), 1, -1)
    img = img * circle[..., None] + 128 * (1 - circle[..., None])
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


class HybridAPTOSDataset(Dataset):
    def __init__(self, dataframe, feature_columns, mean, std, image_size, transform):
        self.df = dataframe.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["img_path"])
        if image is None:
            raise FileNotFoundError(row["img_path"])
        image = self.transform(ben_graham_preprocess(image, self.image_size))
        features = row[self.feature_columns].to_numpy(np.float32)
        features = torch.from_numpy((features - self.mean) / self.std)
        return image, features, torch.tensor(row["diagnosis"], dtype=torch.float32)


class DRHybridModel(nn.Module):
    def __init__(self, engineered_dim, dropout=0.35):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        embedding_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.image_projection = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout)
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(engineered_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.15)
        )
        self.head = nn.Sequential(
            nn.Linear(320, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, images, features):
        image_embedding = self.image_projection(self.backbone(images))
        feature_embedding = self.feature_projection(features)
        return self.head(torch.cat([image_embedding, feature_embedding], 1)).squeeze(1)


class CNNOnlyModel(nn.Module):
    def __init__(self, dropout=0.35):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        embedding_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, images, _features):
        return self.head(self.backbone(images)).squeeze(1)


def make_loaders(train_df, val_df, feature_columns, image_size=300, batch_size=24, num_workers=2):
    mean = train_df[list(feature_columns)].mean().to_numpy(np.float32)
    std = train_df[list(feature_columns)].std().fillna(1.0).to_numpy(np.float32)
    train_transform = T.Compose([
        T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20), T.RandomAffine(0, translate=(0.03, 0.03), scale=(0.9, 1.1)),
        T.ColorJitter(0.08, 0.08, 0.05), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToPILImage(), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = HybridAPTOSDataset(train_df, feature_columns, mean, std, image_size, train_transform)
    val_ds = HybridAPTOSDataset(val_df, feature_columns, mean, std, image_size, val_transform)
    counts = train_df["diagnosis"].value_counts().to_dict()
    weights = train_df["diagnosis"].map(lambda grade: 1.0 / counts[grade]).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), True)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        mean, std,
    )


def apply_thresholds(predictions, thresholds):
    return np.digitize(predictions, np.asarray(thresholds)).astype(np.int64)


def optimize_qwk_thresholds(predictions, labels, initial=(0.5, 1.5, 2.5, 3.5)):
    thresholds = np.asarray(initial, np.float64)
    best = cohen_kappa_score(labels, apply_thresholds(predictions, thresholds), weights="quadratic")
    for width in (0.5, 0.2, 0.08, 0.03):
        for _ in range(8):
            changed = False
            for i in range(4):
                low = thresholds[i - 1] + 0.02 if i else -0.25
                high = thresholds[i + 1] - 0.02 if i < 3 else 4.25
                candidates = np.linspace(max(low, thresholds[i] - width), min(high, thresholds[i] + width), 21)
                for candidate in candidates:
                    trial = thresholds.copy()
                    trial[i] = candidate
                    score = cohen_kappa_score(labels, apply_thresholds(predictions, trial), weights="quadratic")
                    if score > best:
                        thresholds, best, changed = trial, score, True
            if not changed:
                break
    return thresholds.astype(np.float32)

ththt

def train_model(model, train_loader, val_loader, device, checkpoint_path, epochs=25, patience=6):
    model.to(device)
    criterion = nn.SmoothL1Loss(beta=0.75)
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 8e-5},
        {"params": head_params, "lr": 4e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_qwk, stale, best_thresholds = -1.0, 0, np.array([0.5, 1.5, 2.5, 3.5])

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = seen = 0
        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images, features)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)

        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for images, features, labels in val_loader:
                predictions.append(model(images.to(device), features.to(device)).cpu().numpy())
                targets.append(labels.numpy())
        predictions, targets = np.concatenate(predictions), np.concatenate(targets).astype(int)
        thresholds = optimize_qwk_thresholds(predictions, targets)
        qwk = cohen_kappa_score(targets, apply_thresholds(predictions, thresholds), weights="quadratic")
        scheduler.step()
        print(f"epoch={epoch:02d} loss={loss_sum/max(seen,1):.4f} qwk={qwk:.4f} thresholds={thresholds.round(3)}")
        if qwk > best_qwk:
            best_qwk, best_thresholds, stale = qwk, thresholds, 0
            torch.save({"model_state_dict": model.state_dict(), "thresholds": thresholds, "val_qwk": qwk}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return {"best_qwk": best_qwk, "thresholds": best_thresholds, "checkpoint_path": checkpoint_path}


def save_inference_metadata(path, feature_columns: Sequence[str], mean, std, thresholds, image_size):
    Path(path).write_text(json.dumps({
        "feature_columns": list(feature_columns), "feature_mean": mean.tolist(),
        "feature_std": std.tolist(), "thresholds": thresholds.tolist(),
        "image_size": image_size,
    }, indent=2), encoding="utf-8")
