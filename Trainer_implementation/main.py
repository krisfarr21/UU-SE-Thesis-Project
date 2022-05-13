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

# If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'offline'

import datetime


# https://docs.wandb.ai/guides/track/launch#init-start-error



class hf_model():
    # global class variables
    IMPLEM_PROJECT_DIR = '/home/krisfarr/thesis/'
    STORAGE_PROJECT_DIR = '/proj/uppmax2021-2-31'

    LANGUAGES = ['italian', 'arabic', 'english', 'spanish', 'dutch']
    TEST_DF = pd.read_csv(f'{IMPLEM_PROJECT_DIR}final_datasets/maltese_data.csv')

    # pd.read_csv(f'../final_datasets/{lang}_data.csv')
    LABEL_LIST = ['B-PER', 'O', 'I-LOC', 'B-MISC', 'B-ORG', 'B-LOC', 'I-ORG', 'I-PER', 'I-MISC']
    TASK = 'ner'
    BATCH_SIZE = 4 # batch size reduced to 4 due to CUDA out of memory error

    def __init__(self):
        self.argparser()
        # self.model_checkpoint = None
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, 
                                                                num_labels=len(self.LABEL_LIST))
        
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.metric = load_metric("seqeval")

        # source language, mono/multilingual training data, mono/multilingual validation data 
        # self.source_language = None
        # self.training_lingual = None
        # self.validation_lingual = None

        # datasets
        self.train_df = None
        self.valid_df = None
        self.test_df = self.TEST_DF

        # training variables
        self.training_arguments = None
        self.trainer = None

        # directories for outputting results and logs
        self.output_dir = f"{self.STORAGE_PROJECT_DIR}/kris/models/{self.source_language}/{self.model_name}/train_{self.training_lingual}_valid_{self.validation_lingual}"
        self.logging_dir = f'{self.STORAGE_PROJECT_DIR}/kris/loggings/{self.source_language}/{self.model_name}/train_{self.training_lingual}_valid_{self.validation_lingual}'

        # storing results to plot
        self.results = None



    def get_training_data(self) -> pd.DataFrame:
        '''
        
        '''
        if self.training_lingual=='mono':
            self.train_df = pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}/final_datasets/{self.source_language}_data.csv')
        else:
            # get length of source language
            
            source_language_df = pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}/final_datasets/{self.source_language}_data.csv')
            length_source_lang = len(source_language_df)

            print(f"{self.source_language} INITIAL LENGTH: {length_source_lang}")
            # double the number of sentences from source language 
            num_of_sentences_source = int(length_source_lang//len(self.LANGUAGES)) * 2

            # print("NUMBER OF SENTENCES FROM SOURCE", num_of_sentences_source)

            # number of sentences from the other languages (deduct the number of sentences from source from the length of the source
            # and divide it by the number of other languages)
            length_to_split = int((length_source_lang-num_of_sentences_source) // (len(self.LANGUAGES)-1))
            # print(f"NUMBER OF SENTENCES FROM EACH LANGUAGE FOR TRAINING DATA {length_to_split}")
            
            # get the same number of sentences frome each language except for the source language (oversampling it by 100%)
            len_of_dfs ,dfs = [(len(pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}/final_datasets/{lang}_data.csv')[:length_to_split]), lang) for lang in self.LANGUAGES if lang != self.source_language], \
                             [pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}/final_datasets/{lang}_data.csv')[:length_to_split] for lang in self.LANGUAGES if lang != self.source_language]
            dfs.append(source_language_df[:num_of_sentences_source])

            # total_sentences = 0
            # for length, lang in len_of_dfs:
            #     print(f"len of {lang} is {length}")
            #     total_sentences += length
            # print("TOTAL NUMBER OF SENTENCES GATHERED", total_sentences)

            # concatenate the dataframes
            self.train_df = pd.concat(dfs)
        return self.train_df

    def training_validation_data(self, split=0.2) -> pd.DataFrame:
        '''
        '''
        # validation size
        validation_size = int(len(self.train_df) * split)
        if self.validation_lingual=='mono':
            
            
            if self.training_lingual=='mono':
                self.valid_df = self.train_df[:validation_size]
                self.train_df = self.train_df[validation_size:]
            else:
                # get source language data (necessary when having multilingual training data)
                source_language_df = pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}/final_datasets/{self.source_language}_data.csv')
                # validation sentences: extracting the final sentences instead of the starting ones to avoid training/validation overlap
                self.valid_df = source_language_df[len(source_language_df)-validation_size:]
                # reduce size of training data
                self.train_df = self.train_df[validation_size:]
        # self.training_lingual=='multi'
        else:
            # number of sentences per language
            num_of_sentences = int(validation_size // len(self.LANGUAGES))
            print("NUMBER OF SENTENCES FROM EACH LANGUAGE  FOR VALIDATION DATA: ", num_of_sentences)
            if self.training_lingual=='mono':
                # get number of sentences from each language depending on the size of the validation dataset
                
                # remove sentences from training dataframe to keep same percentage split
                # moreover, the sentences are removed to avoid overlapping between training and validation
                self.train_df = self.train_df[validation_size:]

                # get number of sentences needed from each language
                dfs = [pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}final_datasets/{lang}_data.csv')[:num_of_sentences] for lang in self.LANGUAGES]

                #concat dataframes
                self.valid_df = pd.concat(dfs)
            else:
                # remove sentences from traning dataframe
                self.train_df = self.train_df[:len(self.train_df)-validation_size]
                # get number of sentences needed from each language
                dfs = []
                for lang in self.LANGUAGES:
                    df = pd.read_csv(f'{self.IMPLEM_PROJECT_DIR}final_datasets/{lang}_data.csv')
                    if lang=='arabic':
                        print("ARABIC DATASET LENGTH == ", len(df))
                    df = df[len(df)-num_of_sentences:]
                    dfs.append(df)
                self.valid_df = pd.concat(dfs)
        print("Making sure there's no overlap between training and validation:")
        print(self.train_df.merge(self.valid_df, how='inner', indicator=False), '\n')
        return self.train_df, self.valid_df

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert string values to list, shuffle (especially for multilingual data), create id column and convert to Dataset object
        '''
        df['ner_tags'] = df['ner_tags'].apply(eval)
        df['tokens'] = df['tokens'].apply(eval)
        df = df.sample(frac=1).reset_index(drop=True)
        # df = df[(df['tokens'].apply(lambda x: len(x) > 3))]
        df['id'] = [i+1 for i in range(len(df))]
        df = Dataset.from_pandas(df)
        return df


    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        label_all_tokens = True
        labels = []
        for i, label in enumerate(examples[f"{self.TASK}_tags"]):
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
    
    def create_tokenized_datasets(self, df: pd.DataFrame):
        df = df.map(self.tokenize_and_align_labels, batched=True)
        return df

    def remove_special_tokens(self, predictions, labels):
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_predictions, true_labels

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions, true_labels = self.remove_special_tokens(predictions, labels)

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    def train_args(self):
        self.training_arguments = TrainingArguments( \
        output_dir=self.output_dir,
        evaluation_strategy = "epoch",
        logging_strategy = 'epoch',
        save_steps = 5000,
        logging_steps = 5000,
        learning_rate=2e-5,
        per_device_train_batch_size=self.BATCH_SIZE,
        per_device_eval_batch_size=self.BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=True,
        report_to="wandb",
        logging_dir=self.logging_dir)

        return self.training_arguments

    def create_trainer(self):           
        self.trainer = Trainer(
        self.model,
        self.train_args(),
        train_dataset=self.train_df,
        eval_dataset=self.valid_df,
        data_collator=self.data_collator,
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_metrics
        )
        return self.trainer
    
    
    def train_and_eval(self):
        self.trainer.train()
    
    def create_training_stats(self):
        log_history = self.trainer.state.log_history
        train_loss = [log['loss'] for log in log_history if 'loss' in log.keys()]
        eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log.keys()]
        training_stats = {'Training Loss': train_loss,
                  'Evaluation Loss': eval_loss,
                  'Epochs': [i+1 for i in range(self.training_arguments.num_train_epochs)]}
        return training_stats

    def plot_stats(self):
        training_stats = self.create_training_stats()
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('Epochs')

        
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Evaluation Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3])

        plt.savefig(f"{self.STORAGE_PROJECT_DIR}/kris/results/{self.source_language}/{self.model_checkpoint}/train_eval_stats-train_{self.training_lingual}-valid_{self.validation_lingual}.png")
        plt.show()
    
    def predict(self):
        self.test_df = self.create_tokenized_datasets(df=self.test_df)
        # self.train_df.map(self.tokenize_and_align_labels, batched=True)
        predictions, labels, _ = self.trainer.predict(self.test_df)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions, true_labels = self.remove_special_tokens(predictions, labels)
        
        self.results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return self.results   


    def save_results(self):
        # ## RESULTS INTO CSV FILE
        tags_results = {k:v for (k,v) in self.results.items() if 'overall' not in k}
        overall_results = {k:[v] for (k,v) in self.results.items() if 'overall' in k}


        tags_results_pd = pd.DataFrame.from_dict(tags_results)[:3]
        overall_results_pd = pd.DataFrame.from_dict(overall_results, orient='columns')[:3]

        tags_results_pd.to_csv(f'{self.STORAGE_PROJECT_DIR}/kris/results/{self.source_language}/{self.model_checkpoint}/tags_results-train_{self.training_lingual}-valid_{self.validation_lingual}.csv')
        plot = tags_results_pd.plot.bar(fontsize=12, ylim=(0, 1))

        fig = plot.get_figure()
        fig.savefig(f"{self.STORAGE_PROJECT_DIR}/kris/results/{self.source_language}/{self.model_checkpoint}/tags_results-train_{self.training_lingual}-valid_{self.validation_lingual}.png")
    

    def argparser(self):
        
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-lang', '--language', type=str, choices=['italian', 'arabic', 'english', 'spanish', 'dutch'], 
                            help="choose source lang", required=True)
        parser.add_argument('--model_checkpoint', type=str, choices=['mbert', 'xlm-r'], help="choose mbert or xlm-r", required=True)
        parser.add_argument('-tr-ling','--training_lingual', type=str, choices=['mono', 'multi'], 
                            help='Whether training data is mono or multi lingual', required=True)
        parser.add_argument('-valid-ling', '--validation_lingual',type=str, choices=['mono', 'multi'], 
                            help='Whether training data is mono or multi lingual', required=True)
        # parser.add_argument('--model_checkpoint', type=str, choices=['xlm-r', 'mbert'], help='choose between multilingual bert and xlm-roberta')
        args = parser.parse_args()
        
        if args.model_checkpoint == 'mbert':
            self.model_checkpoint = 'bert-base-multilingual-cased'
        elif args.model_checkpoint == 'xlm-r':
            self.model_checkpoint = 'xlm-roberta-base'
        print("#" * 50)
        print("#" * 50, '\n')
        print("USING: ", self.model_checkpoint, '\n')

        self.source_language = args.language
        print("SOURCE LANGUAGE ", self.source_language, '\n')

        self.training_lingual = args.training_lingual
        print("TRAINING IS: ", self.training_lingual, '\n')

        self.validation_lingual = args.validation_lingual
        print("VALIDATION IS: ", self.validation_lingual, '\n')
        print("#" * 50)
        print("#" * 50)

        wandb.init(settings=wandb.Settings(start_method="fork"), project='thesis-'+self.model_checkpoint, 
                    name=self.source_language + '-' + self.training_lingual + '-' + self.validation_lingual)

        
    def main(self):
        print("<<<< TRAINING AND VALIDAITON DATA >>>>>")
        self.valid_df = self.get_training_data()
        self.train_df, self.valid_df = self.training_validation_data()
            
        print('\n\n, <<<<< Preprocessing the datasets >>>>>')

        print("\n\n <<<<< Length of dataset before preprocessing >>>>>")
        print(f"Length of training {len(self.train_df)}",  f"Length of validation {len(self.valid_df)}", f"Length of testing {len(self.test_df)}")
        
        self.train_df, self.valid_df = self.preprocess_dataframe(self.train_df), self.preprocess_dataframe(self.valid_df)
        self.test_df = self.preprocess_dataframe(self.test_df)

        print("\n\n, <<<<< Tokenizing datasets >>>>>")

        self.train_df, self.valid_df = self.create_tokenized_datasets(df=self.train_df), self.create_tokenized_datasets(df=self.valid_df)
        self.test_df = self.create_tokenized_datasets(df=self.test_df) 

         

        print("\n\n <<<<< Length of datasets after preprocessing >>>>>")
        print(f"Length of training {len(self.train_df)}",  f"Length of validation {len(self.valid_df)}", f"Length of testing {len(self.test_df)}") 

        print("\n\n")
        self.trainer = self.create_trainer()
        self.train_and_eval()
        self.plot_stats()

        self.predict()
        self.save_results()

if __name__ == '__main__':
    start = datetime.datetime.now()
    print(f"Job started at: {str(start)}")

    xlm_r_model = hf_model()
    xlm_r_model.main()

    end = datetime.datetime.now()
    print("#" * 50)
    print(f"Job finished at {str(end)} and files were saved")
    print("#" * 50, "\n \n")

    # print("MULTILINGUAL BERT \n")c
    # print("#" * 50, '\n')
    # mbert_model = hf_model(model_checkpoint='bert-base-multilingual-cased')
    # mbert_model.main()
    # print("#" * 50, '\n')
    # print("JOB FINISHED AND FILED SAVED")
    # print("#" * 50, '\n')