import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy

from helpers.txt_helper import clear_text_for_classification


class DataForClassification:
    def __init__(self, conf):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.tokenizer = BertTokenizer.from_pretrained(conf.PATH_TO_PRETRAINED_BERT_MODEL, do_lower_case=False)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        self.datapath = f'{conf.PATH_TO_DATASET_CATALOG}/{conf.TRAIN_FILENAME}'
        #self.datapath = f'{conf["PATH_TO_DATASET_CATALOG"]}/{conf["TRAIN_FILENAME"]}'

        self.batch_size = 100

    def read_data(self):
        #df = pd.read_csv(self.datapath, delimiter=',', header=None, usecols=[1, 2], names=['label', 'sentence'], skiprows=1, nrows=100)
        df = pd.read_csv(self.datapath, delimiter=',', header=None, usecols=[1, 2], names=['label', 'sentence'], skiprows=1)
        #df = pd.read_csv(self.datapath, delimiter=',', usecols=['topics', 'title'], nrows=20)
        print(df.head)

        #df.to_csv(self.datapath, sep=',', encoding='utf-8')

        sentences = df.sentence.str.strip().values
        labels = df.label.str.strip().values

        label_dict, ids_to_label = self.create_label_dict(labels)

        labels_ids = []
        for label in labels:
            labels_ids.append(label_dict[label])

        max_len = 0

        sentences = [clear_text_for_classification(sent) for sent in sentences]

        for sent in sentences:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:

            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels_ids = torch.tensor(labels_ids)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels_ids)

        # Calculate the number of samples to include in each set.
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=self.batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )


        test_dataloader = DataLoader(
            test_dataset,  # The validation samples.
            sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )

        return label_dict, train_dataloader, validation_dataloader, test_dataloader, max_len


    def create_label_dict(self, txt_label):
        unique_labels = list(set(txt_label))

        labels_to_ids = {k: v for v, k in enumerate(unique_labels)}

        ids_to_labels = {v: k for k, v in labels_to_ids.items()}

        return labels_to_ids, ids_to_labels

    def get_class_distribution(self):
        df = pd.read_csv(self.datapath, delimiter=',', header=None, names=['sentence', 'label'], skiprows=1)

        labels = df.label.str.strip().values

        unique, counts = numpy.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))