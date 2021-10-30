import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from helpers.txt_helper import clear_text_for_classification
import numpy as np

class Classify:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not Classify._instance:
            Classify._instance = super(Classify, cls).__new__(cls)

        return Classify._instance

    def __init__(self, conf):
        if not hasattr(self, 'is_init'):
            self.is_init = True

            #self.path_to_pretrained_bert = conf["PATH_TO_PRETRAINED_BERT_MODEL"]
            self.path_to_saved_model = conf["PATH_TO_SAVED_MODEL"]

            self.device = None
            self.get_device()

            self.map_location = None
            self.get_map_location()

            print(f'Loading pytorch classifying model...')
            self.checkpoint = torch.load(self.path_to_saved_model + '/saved_model.pt', map_location=self.map_location)


            print(f'The pytorch classifying model was loading')
            print(f'Epoch: {self.checkpoint["epoch"]}')
            print(f'Validation loss: {self.checkpoint["val_acc"]}')

            self.batch_size = 32

            print('Loading pytorch model from bert pretrained ...')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-multilingual-cased',  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=len(self.checkpoint['labels_dict']),
                # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )

            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.model.eval()

            print('Loading BERT tokenizer...')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', truncation=True)

        pass

    def get_device(self):
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))

            self.device = torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def get_map_location(self):
        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = 'cpu'

    def predict(self, user_input):

        if self.device.type == 'cuda':
            self.model.cuda()
        else:
            self.model.cpu()

        ids_to_labels = {v: k for k, v in self.checkpoint["labels_dict"].items()}

        pure_input_message = clear_text_for_classification(user_input)

        data_dict = self.get_data_from_input(pure_input_message, self.checkpoint['max_len'])

        # Forward pass, calculate logit predictions

        outputs = self.model(data_dict['data_text_ids'].to(self.device), token_type_ids=None, attention_mask=data_dict['attention_masks'].to(self.device))

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        pred_labels_i = np.argmax(logits, axis=1).flatten()

        softmaxFunc = torch.nn.Softmax(dim=1)
        softmaxScores = softmaxFunc(outputs[0])

        prob = np.max(softmaxScores.detach().cpu().numpy())

        return ids_to_labels[pred_labels_i[0]], round(prob * 100, 1)

    def get_data_from_input(self, input_text, max_len):
        data_dict = {'sentences': [input_text]}
        # Create sentence and label lists

        encoded_dict = self.tokenizer.encode_plus(
            input_text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )

        data_dict['data_text_ids'] = encoded_dict['input_ids']
        data_dict['attention_masks'] = encoded_dict['attention_mask']

        return data_dict

