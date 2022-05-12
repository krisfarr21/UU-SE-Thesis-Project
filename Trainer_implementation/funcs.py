import transformers
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
import numpy as np
# from monolingual_split import *

LABEL_LIST = ['B-PER', 'O', 'I-LOC', 'B-MISC', 'B-ORG', 'B-LOC', 'I-ORG', 'I-PER', 'I-MISC']

def split_datasets(source_lang, valid_split=0.2):
    train_df = pd.read_csv(f"../final_datasets/{source_lang}_data.csv")[:500]
    print("LENGTH---->", len(train_df))
    split_size = int(len(train_df) * valid_split)
    valid_df = train_df[:split_size]
    train_df = train_df[split_size:]
    test_df = pd.read_csv("../final_datasets/maltese_data.csv")
    train_dataset, valid_dataset, test_dataset = \
    Dataset.from_pandas(train_df), \
    Dataset.from_pandas(valid_df), \
    Dataset.from_pandas(test_df)
    return train_dataset, valid_dataset, test_dataset

def tokenize_and_align_labels(examples, tokenizer, task):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    label_all_tokens = True
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    metric = load_metric("seqeval")
    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }