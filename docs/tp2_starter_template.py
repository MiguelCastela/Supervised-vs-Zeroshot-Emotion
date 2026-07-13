import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

# check TP2 assignment instructions for links to other model options
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE_EMB = 64
BATCH_SIZE_TRAIN = 64
EPOCHS = 5
LR = 1e-3
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


class ANNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
    model = SentenceTransformer(model_name)
    return model


# ---------------------------------------------------------
# 3) Compute embeddings
# ---------------------------------------------------------

def compute_embeddings(model, X_train_texts, X_val_texts, X_test_texts):
    X_train = model.encode(
        X_train_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    X_val = model.encode(
        X_val_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    X_test = model.encode(
        X_test_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"\nEmbeddings shapes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    return X_train, X_val, X_test


# ---------------------------------------------------------
# 4) Prepare PyTorch loaders
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 5) Train ANN classifier
# ---------------------------------------------------------

def train_ann_classifier(input_dim, num_classes, train_loader, val_loader, device):
    ann = ANNClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining ANN classifier...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # Training
        ann.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = ann(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        ann.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = ann(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                true.extend(yb.numpy().tolist())

        val_acc = accuracy_score(true, preds)
        print(f"Epoch {epoch}: val acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = ann.state_dict()

    # load best model (optional but usually good practice)
    if best_state is not None:
        ann.load_state_dict(best_state)

    return ann


# ---------------------------------------------------------
# 6) Evaluate on test set
# ---------------------------------------------------------

def evaluate_on_test(model, test_loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            true.extend(yb.numpy().tolist())

    acc = accuracy_score(true, preds)
    cm = confusion_matrix(true, preds)
    f1 = f1_score(true, preds, average='macro')

    print("\n========== Test Results ==========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    return acc, cm


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    set_seed(SEED) # tem de ser o mesmo para parte 1 e parte 2

    # 1) Load dataset
    X_train_texts,y_train,X_val_texts,y_val,X_test_texts,y_test,label_names,num_classes = load_emotion_dataset()
    print(f"Labels: {label_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(X_train_texts)}")
    print(f"Number of validation samples: {len(X_val_texts)}")
    print(f"Number of test samples: {len(X_test_texts)}")
    print(f"Example text: {X_train_texts[0]}    Label: {label_names[y_train[0]]}")

    ### Part 1
    # 2) Load embedding model
    emb_model = load_embedding_model(MODEL_NAME)

    # 3) Compute embeddings
    X_train, X_val, X_test = compute_embeddings(emb_model, X_train_texts, X_val_texts, X_test_texts)
    input_dim = X_train.shape[1]

    # para fazer comparações podemos gravar os embeddings em numpy ou qq coisa desse genero



    # 4) Prepare PyTorch loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    # 5) Train ANN classifier
    ann = train_ann_classifier(input_dim, num_classes, train_loader, val_loader, DEVICE)

    # 6) Evaluate on test set
    evaluate_on_test(ann, test_loader, DEVICE)

    #macro F1 score
    

    #onde eatá a validação???

    ### end part 1



if __name__ == "__main__":
    main()
