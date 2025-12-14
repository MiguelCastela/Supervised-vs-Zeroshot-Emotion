from dataloader import SEED, set_seed, load_emotion_dataset, load_embedding_model, prepare_dataloaders, DEVICE, MODEL_NAME 
from transformer_classifier import compute_embeddings, train_ann_classifier, evaluate_on_test, train_logreg_classifier, evaluate_sklearn_classifier
from NLI import part_two as run_nli_part_two
import numpy as np
import argparse
import os

N_SEEDS = 5


def part_one(head_type: str = "ann"):

    emb_model = load_embedding_model(MODEL_NAME)



    X_train, X_val, X_test = compute_embeddings(emb_model, X_train_texts, X_val_texts, X_test_texts)

    np.save("embeddings_train.npy", X_train)
    np.save("embeddings_val.npy", X_val)
    np.save("embeddings_test.npy", X_test)
    np.save("labels_train.npy", y_train)
    np.save("labels_val.npy", y_val)
    np.save("labels_test.npy", y_test)

    input_dim = X_train.shape[1]

    # Aggregate metrics over seeds
    accs = []
    f1s = []
    cms = []

    for i in range(N_SEEDS):
        current_seed = SEED + i
        set_seed(current_seed)

        # 4) Prepare PyTorch loaders (same embeddings, different shuffles)
        train_loader, val_loader, test_loader = prepare_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        if head_type == "logreg":
            clf = train_logreg_classifier(X_train, y_train)
            acc, f1, cm = evaluate_sklearn_classifier(clf, X_test, y_test)
        else:
            ann = train_ann_classifier(input_dim, num_classes, train_loader, val_loader)
            acc, f1, cm = evaluate_on_test(ann, test_loader)
        accs.append(acc)
        f1s.append(f1)
        cms.append(cm)

    print("\n========== Mean over seeds ==========")
    print(f"Seeds: {N_SEEDS}, base SEED: {SEED}")
    print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Confusion matrix aggregation (counts)
    cms_arr = np.stack(cms, axis=0)  # (N_SEEDS, C, C)
    mean_cm = cms_arr.mean(axis=0)
    std_cm = cms_arr.std(axis=0)

    print("\nMean Confusion Matrix (counts):")
    print(mean_cm.round(2))
    print("Std Confusion Matrix (counts):")
    print(std_cm.round(2))

    # Row-normalized confusion matrices
    # Avoid division by zero for classes with zero support
    row_sums = cms_arr.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cms_norm = np.divide(cms_arr, row_sums, where=row_sums!=0)
        cms_norm = np.nan_to_num(cms_norm)

    mean_cm_norm = cms_norm.mean(axis=0)
    std_cm_norm = cms_norm.std(axis=0)

    print("\nMean Confusion Matrix (row-normalized):")
    print(mean_cm_norm.round(3))
    print("Std Confusion Matrix (row-normalized):")
    print(std_cm_norm.round(3))
    

def part_two():
    """Run NLI zero-shot classification (current `NLI.py` main) from main.py."""
    # Using DeBERTa v3 large zero-shot model
    model_name = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    acc_nli, f1_nli, cm_nli, _ = run_nli_part_two(model_name=model_name)
    print("\n========== Part 2: NLI (Zero-Shot) ==========")
    print(f"Model: {model_name}")
    print(f"Accuracy:  {acc_nli:.4f}")
    print(f"Macro F1:  {f1_nli:.4f}")
    print("Confusion matrix (counts):")
    print(cm_nli)

    # Match Part 1 confusion matrix logic: show row-normalized matrix
    import numpy as np
    row_sums = cm_nli.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_nli_norm = np.divide(cm_nli, row_sums, where=row_sums!=0)
        cm_nli_norm = np.nan_to_num(cm_nli_norm)

    print("Confusion matrix (row-normalized):")
    print(cm_nli_norm.round(3))


if __name__ == "__main__":

    set_seed(SEED) # tem de ser o mesmo para parte 1 e parte 2

    # 1) Load dataset
    X_train_texts,y_train,X_val_texts,y_val,X_test_texts,y_test,label_names,num_classes = load_emotion_dataset()
    print(f"Labels: {label_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(X_train_texts)}")
    print(f"Number of validation samples: {len(X_val_texts)}")
    print(f"Number of test samples: {len(X_test_texts)}")
    print(f"Example text: {X_train_texts[0]}    Label: {label_names[y_train[0]]}")

    #part_one(head_type="logreg")
    part_two()
    

    

