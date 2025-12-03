import torch
import torch.nn as nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from dataloader import BATCH_SIZE_TRAIN, EPOCHS, LR, DEVICE, BATCH_SIZE_EMB

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
