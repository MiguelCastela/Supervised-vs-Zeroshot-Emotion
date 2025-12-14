import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

# check TP2 assignment instructions for links to other model options
#MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE_TRAIN = 64
SEED = 2025


## ubuntu or windows with cuda support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## macos with M series support
# DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------------

def load_emotion_dataset():
    ds = load_dataset("dair-ai/emotion")
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    label_names = train_ds.features["label"].names
    num_classes = len(label_names)

    X_train_texts = train_ds["text"]
    y_train = np.array(train_ds["label"])
    X_val_texts = val_ds["text"]
    y_val = np.array(val_ds["label"])
    X_test_texts = test_ds["text"]
    y_test = np.array(test_ds["label"])

    return X_train_texts, y_train, X_val_texts, y_val, X_test_texts, y_test, label_names, num_classes


# ---------------------------------------------------------
# 2) Load embedding model
# ---------------------------------------------------------

def load_embedding_model(model_name: str):
    print(f"\nLoading model: {model_name}")
    try:
        # Prefer initializing on the configured DEVICE; SentenceTransformer accepts a device string
        model = SentenceTransformer(model_name, device=str(DEVICE))
        return model
    except Exception as e:
        print(f"Warning: failed to load model on {DEVICE} ({e}). Falling back to CPU.")
        model = SentenceTransformer(model_name, device="cpu")
        return model


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

