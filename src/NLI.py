import torch
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from dataloader import load_emotion_dataset, set_seed, SEED, DEVICE

def run_zero_shot(texts, labels, model_name="facebook/bart-large-mnli", batch_size=32, template="This text expresses {}."):
    """
    Perform zero-shot classification on a list of texts.
    
    Args:
        texts (list): List of text samples to classify.
        labels (list): List of candidate labels (e.g., ["joy", "sadness", ...]).
        model_name (str): Name of the NLI model to use.
        batch_size (int): Batch size for inference.
        template (str): Hypothesis template to use.
        
    Returns:
        list: Predicted labels.
    """
    print(f"Loading model {model_name} on {DEVICE}...")
    
    # Initialize the pipeline
    # Note: pipeline handles device placement. 
    # If DEVICE is a torch.device, we can pass it directly or pass the device index.
    classifier = pipeline("zero-shot-classification", model=model_name, device=DEVICE)
    
    preds = []
    
    print(f"Classifying {len(texts)} texts...")
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch = texts[i:i+batch_size]
        
        # The pipeline returns a list of dictionaries
        # Each dictionary contains 'sequence', 'labels', 'scores'
        # 'labels' are sorted by score (descending)
        batch_results = classifier(batch, candidate_labels=labels, hypothesis_template=template)
        
        for res in batch_results:
            # The first label in the list is the highest scoring one
            preds.append(res['labels'][0])
            
    return preds

if __name__ == "__main__":
    set_seed(SEED)
    
    # 1. Load Dataset
    # We only need the test set for this part, as Zero-Shot doesn't require training
    print("Loading dataset...")
    _, _, _, _, X_test_texts, y_test, label_names, _ = load_emotion_dataset()
    
    print(f"Labels: {label_names}")
    print(f"Test set size: {len(X_test_texts)}")
    
    # 2. Run Zero-Shot Classification
    # We use a hypothesis template suitable for emotion detection
    template = "This text expresses {}."
    
    # You can experiment with different models, e.g.:
    # - "facebook/bart-large-mnli" (High accuracy, slower)
    # - "cross-encoder/nli-distilroberta-base" (Faster, might be less accurate for zero-shot pipeline usage directly without adaptation)
    # - "typeform/distilbert-base-uncased-mnli"
    model_name = "facebook/bart-large-mnli"
    
    predicted_labels = run_zero_shot(X_test_texts, label_names, model_name=model_name, template=template)
    
    # 3. Evaluate
    # Convert numeric ground truth to string labels
    true_labels = [label_names[i] for i in y_test]
    
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nZero-Shot Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, digits=4))
    
    # Print Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=label_names)
    print("\nConfusion Matrix:")
    print(cm)
