import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from dataloader import DEVICE


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE_EMB = 64
EPOCHS = 25
LR = 1e-3
SEED = 2025
EARLY_STOPPING_PATIENCE = 3

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


def train_ann_classifier(input_dim, num_classes, train_loader, val_loader, device: torch.device = DEVICE):
    ann = ANNClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining ANN classifier...")
    
    # Initialize Early Stopping variables
    best_val_f1 = -1.0
    patience_counter = 0
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
        # Calculate Macro F1 Score for Early Stopping
        val_f1 = f1_score(true, preds, average='macro')
        
        print(f"Epoch {epoch}: val acc = {val_acc:.4f}, val f1 = {val_f1:.4f}")

        # Early Stopping Logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = ann.state_dict() # Save best model state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best Macro F1: {best_val_f1:.4f}")
                break

    # Load best model state before returning
    if best_state is not None:
        ann.load_state_dict(best_state)
    else:
        # Fallback in case no improvement was ever made (e.g., if best_val_f1 starts at -1 and validation is 0)
        # This is unlikely but ensures a model is returned.
        print("Note: No improved state found or only one epoch ran. Returning last epoch's model.")


    return ann


def evaluate_on_test(model, test_loader, device: torch.device = DEVICE):
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

    return acc, f1, cm


def train_logreg_classifier(X_train, y_train):
    """Train a multinomial Logistic Regression head on frozen embeddings.

    Uses lbfgs solver with multiclass='multinomial'. Increase max_iter for convergence.
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_sklearn_classifier(model, X_test, y_test):
    """Evaluate a scikit-learn classifier on numpy embeddings, returning (acc, f1, cm)."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    cm = confusion_matrix(y_test, preds)
    print("\n========== Test Results (LogReg) ==========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    return acc, f1, cm
