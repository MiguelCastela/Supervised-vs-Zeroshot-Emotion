from dataloader import SEED, set_seed, load_emotion_dataset, load_embedding_model, prepare_dataloaders, DEVICE, MODEL_NAME 
from transformer_classifier import compute_embeddings, train_ann_classifier, evaluate_on_test

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
