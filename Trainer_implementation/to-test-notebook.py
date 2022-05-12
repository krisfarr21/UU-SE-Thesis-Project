import transformers
from datasets import load_dataset, load_metric, Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly

import seaborn as sns
import argparse
import wandb

from transformers import AutoTokenizer, default_data_collator

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from transformers import DataCollatorForTokenClassification

# https://github.com/ThilinaRajapakse/simpletransformers/issues/515
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# https://docs.wandb.ai/guides/track/launch#init-start-error
wandb.init(settings=wandb.Settings(start_method="fork"), project='thesis')

LANGUAGE = ['italian', 'arabic', 'english', 'spanish', 'dutch']
TEST_DF = pd.read_csv('../final_datasets/maltese_data.csv')
LABEL_LIST = ['B-PER', 'O', 'I-LOC', 'B-MISC', 'B-ORG', 'B-LOC', 'I-ORG', 'I-PER', 'I-MISC']
TASK = 'ner'
BATCH_SIZE = 4 # batch size reduced to 4 due to CUDA out of memory error

MODEL_CHECKPOINT = 'bert-base-multilingual-cased' # OR 'xlm-roberta-base'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
assert isinstance(TOKENIZER, transformers.PreTrainedTokenizerFast)

MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=len(LABEL_LIST))



DATA_COLLATOR = DataCollatorForTokenClassification(TOKENIZER)
METRIC = load_metric("seqeval")

MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]


def get_training_data(language: str, training_data_lingual:str) -> pd.DataFrame:
    '''
    
    '''
    if training_data_lingual=='mono':
        training_df = pd.read_csv(f'../final_datasets/{language}_data.csv')
    else:
        # every csv file to pandas dataframe
        dfs = [pd.read_csv(f'../final_datasets/{lang}_data.csv') for lang in LANGUAGE]
        # the shortest length amongst the dataframe
        min_len = min([len(df) for df in dfs])
        # indexing the dataframe to avoid under/over sampling
        dfs = [df[:min_len] for df in dfs]
        # concatenating dataframes
        training_df = pd.concat(dfs)
    
    return training_df

def training_validation_data(training_dataframe: pd.DataFrame, validation_data_lingual: str, split=0.2) -> pd.DataFrame:
    '''
    '''
    if validation_data_lingual=='mono':
        train_split_size = int(len(training_dataframe) * split)
        validation_df = training_dataframe[:train_split_size]
        training_df = training_dataframe[train_split_size:]
    else: 
        # get number of sentences from each language depending on the size of the validation dataset
        num_of_rows = int((len(training_dataframe) * split) // 5)
        dfs = [pd.read_csv(f'../final_datasets/{lang}_data.csv')[:num_of_rows] for lang in LANGUAGE]
        # concatenating dataframes
        validation_df = pd.concat(dfs)
        training_df = training_dataframe
    return training_df, validation_df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Convert string values to list, shuffle (especially for multilingual data), create id column and convert to Dataset object
    '''
    df['ner_tags'] = df['ner_tags'].apply(eval)
    df['tokens'] = df['tokens'].apply(eval)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[(df['tokens'].apply(lambda x: len(x) > 3))]
    df['id'] = [i+1 for i in range(len(df))]
    df = Dataset.from_pandas(df)
    return df


def tokenize_and_align_labels(examples):
    tokenized_inputs = TOKENIZER(examples["tokens"], truncation=True, is_split_into_words=True)
    label_all_tokens = True
    labels = []
    for i, label in enumerate(examples[f"{TASK}_tags"]):
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
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang', '--language', type=str, choices=['italian', 'arabic', 'english', 'spanish', 'dutch'], 
                        help="choose source lang", required=True)
    parser.add_argument('-tr-ling','--training_lingual', type=str, choices=['mono', 'multi'], 
                        help='Whether training data is mono or multi lingual', required=True)
    parser.add_argument('-valid-ling', '--validation_lingual',type=str, choices=['mono', 'multi'], 
                        help='Whether training data is mono or multi lingual', required=True)
    # parser.add_argument('--model_checkpoint', type=str, choices=['xlm-r', 'mbert'], help='choose between multilingual bert and xlm-roberta')
    args = parser.parse_args()

    train_df = get_training_data(language=args.language, training_data_lingual=args.training_lingual)
    train_df, valid_df = training_validation_data(train_df, validation_data_lingual=args.validation_lingual)
    test_df = TEST_DF
    train_dataset, valid_dataset, test_dataset = preprocess_dataframe(train_df), preprocess_dataframe(valid_df), preprocess_dataframe(test_df)
    
    print("Tokenizing inputs...")
    train_tokenized_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    valid_tokenized_dataset = valid_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    trainer_args = TrainingArguments(
    output_dir=f"/proj/uppmax2020-2-2/kris/models/{MODEL_NAME}-finetuned-{TASK}-{args.language}-train_{args.training_lingual}-valid_{args.validation_lingual}",
    evaluation_strategy = "epoch",
    logging_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    remove_unused_columns=True,
    report_to="wandb",
    logging_dir=f'/proj/uppmax2020-2-2/kris/loggings/{MODEL_NAME}-finetuned-{TASK}-{args.language}-train_{args.training_lingual}-valid_{args.validation_lingual}'
        )

    trainer = Trainer(
    MODEL,
    trainer_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=valid_tokenized_dataset,
    data_collator=DATA_COLLATOR,
    tokenizer=TOKENIZER,
    compute_metrics=compute_metrics
    )
    # print(trainer)

if __name__ == '__main__':
    main()










# trainer.train()

# log_history = trainer.state.log_history
# log_history_epochs = [log for log in log_history if 'eval_loss' in log.keys()][:-1]
# train_runtime = [log['train_runtime'] for log in log_history if 'train_runtime' in log.keys()][0]
# train_loss = [log['loss'] for log in log_history if 'loss' in log.keys()]
# eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log.keys()]
# training_stats = {'Training Loss': train_loss,
#                   'Evaluation Loss': eval_loss,
#                   'Epochs': [i+1 for i in range(args.num_train_epochs)]}


# # Create a DataFrame from our training statistics.
# df_stats = pd.DataFrame(data=training_stats)

# # Use the 'epoch' as the row index.
# df_stats = df_stats.set_index('Epochs')


# # Use plot styling from seaborn.
# sns.set(style='darkgrid')

# # Increase the plot size and font size.
# sns.set(font_scale=1.5)
# plt.rcParams["figure.figsize"] = (12,6)

# # Plot the learning curve.
# plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
# plt.plot(df_stats['Evaluation Loss'], 'g-o', label="Validation")

# # Label the plot.
# plt.title("Training & Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.xticks([1, 2, 3])

# plt.savefig(f"/proj/uppmax2020-2-2/kris/results/{lang}/train_eval_stats_{model_checkpoint}-monolingual.png")
# plt.show()


# predictions, labels, _ = trainer.predict(test_tokenized_dataset)
# predictions = np.argmax(predictions, axis=2)

# # Remove ignored index (special tokens)
# true_predictions = [
#     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]
# true_labels = [
#     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]

# results = metric.compute(predictions=true_predictions, references=true_labels)
# print(results)   


# ## RESULTS INTO CSV FILE
# tags_results = {k:v for (k,v) in results.items() if 'overall' not in k}
# overall_results = {k:[v] for (k,v) in results.items() if 'overall' in k}


# tags_results_pd = pd.DataFrame.from_dict(tags_results)[:3]
# overall_results_pd = pd.DataFrame.from_dict(overall_results, orient='columns')[:3]

# tags_results_pd.to_csv(f'/proj/uppmax2020-2-2/kris/results/{lang}/tags_results_{model_checkpoint}-monolingual.csv')
# plot = tags_results_pd.plot.bar(fontsize=12, ylim=(0, 1))

# fig = plot.get_figure()
# fig.savefig(f"/proj/uppmax2020-2-2/kris/results/{lang}/tags_results_{model_checkpoint}-monolingual.png")
