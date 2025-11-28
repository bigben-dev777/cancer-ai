import os

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import xgboost as xgb
from PIL import Image
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0


# Step 1: Load metadata and image paths from CSV
def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    # Assume classes are strings, encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Class"])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Encode metadata
    scaler = StandardScaler()
    df["Age_norm"] = scaler.fit_transform(df[["Age"]])

    gender_map = {"m": 0, "f": 1}
    df["Gender_enc"] = df["Gender"].map(gender_map)

    location_dummies = pd.get_dummies(df["Location"], prefix="Location")
    df = pd.concat([df, location_dummies], axis=1)

    meta_columns = ["Age_norm", "Gender_enc"] + list(location_dummies.columns)
    meta_features = df[meta_columns].values.astype(np.float32)

    # Train-test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    return train_df, test_df, meta_features, le, num_classes, meta_columns


# Custom Dataset for images
class SkinDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]["NewFileName"])
        image = Image.open(img_name).convert("RGB")
        label = self.df.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


# Step 2: Image augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# TTA transforms (several augmentations for inference)
tta_transforms = [
    val_transform,  # original
    transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
]


# Step 3: Load ONNX model for inference
def load_onnx_model(onnx_path):
    return ort.InferenceSession(onnx_path)


# Step 4: Inference with TTA
def extract_features_with_tta(image_path, meta, session, tta_transforms):
    image = Image.open(image_path).convert("RGB")
    features_list = []
    for transform in tta_transforms:
        img = transform(image).unsqueeze(0).numpy()  # To numpy for ONNX
        features = session.run(None, {"image": img, "meta": meta})
        features_list.append(features)
    return features


# Extract features for dataset
def extract_dataset_features(df, image_dir, session, tta_transforms, use_tta=True):
    features = []
    for idx in range(len(df)):
        raw = df.iloc[idx]
        img_path = os.path.join(image_dir, raw["NewFileName"])
        meta = {
            "age": raw["Age_norm"],
            "gender": raw["Gender_enc"],
            "location": raw["Location"],
        }
        print(meta)
        print("|" * 60)
        if use_tta:
            feat = extract_features_with_tta(img_path, meta, session, tta_transforms)
        else:
            image = (
                val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).numpy()
            )
            feat = session.run(None, {"input": image})[0]
        features.append(feat.flatten())
    return np.array(features)


# Step 5: Combine results (features + metadata)
def combine_features(features, meta_features):
    return np.hstack((features, meta_features))


# Step 6: Train XGBoost and save to ONNX
def train_xgboost(train_features, train_labels, num_classes, onnx_path="xgboost.onnx"):
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss"
    )
    model.fit(train_features, train_labels)

    input_dim = train_features.shape[1]
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"XGBoost saved to {onnx_path}")
    return model


# Main function
if __name__ == "__main__":
    csv_path = "/home/mateo/safescan/cancer-ai/manager/dataset/data.csv"  # Replace with your CSV path
    image_dir = "/home/mateo/safescan/cancer-ai/manager/dataset/milk10k"  # Replace with your image directory
    onnx_path = "/home/mateo/safescan/cancer-ai/manager/models/2025-11-27/grose/skin_bmodel01.onnx"

    train_df, test_df, meta_features, le, num_classes, meta_columns = load_data(
        csv_path, image_dir
    )

    # Load ONNX
    session = load_onnx_model(onnx_path)

    # Extract features (without TTA for training to save time)
    train_features = extract_dataset_features(
        train_df, image_dir, session, tta_transforms
    )
    test_features = extract_dataset_features(
        test_df, image_dir, session, tta_transforms
    )

    # For demonstration, extract with TTA on test
    # test_features_tta = extract_dataset_features(test_df, image_dir, session, tta_transforms, use_tta=True)

    # Split meta accordingly
    train_idx = train_df.index
    test_idx = test_df.index
    train_meta = meta_features[train_idx]
    test_meta = meta_features[test_idx]

    # Combine
    train_combined = combine_features(train_features, train_meta)
    test_combined = combine_features(test_features, test_meta)

    # Labels
    train_labels = train_df["label"].values
    test_labels = test_df["label"].values

    # Train XGBoost
    xgb_model = train_xgboost(train_combined, train_labels, num_classes)

    # For inference example (using TTA on test)
    probs = xgb_model.predict_proba(test_combined)
    print("|" * 60)
    print(probs)
