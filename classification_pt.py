from data_for_classification import DataForClassification
from classification_config import conf_pt as default_conf
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

class ClassificationPt:
    class DictToObject(object):
        def __init__(self, d):
            self.__dict__ = d

    def __init__(self, conf=default_conf):
        self.conf = self.DictToObject(conf)

        data = DataForClassification(self.conf)

        self.device = data.device

        self.label_dict, self.train_dataloader, self.validation_dataloader, self.test_dataloader, self.max_len = data.read_data()

        self.model = BertForSequenceClassification.from_pretrained(
            #f"{self.conf.PATH_TO_PRETRAINED_BERT_MODEL}/",  # Use the 12-layer BERT model, with an uncased vocab.
            'bert-base-multilingual-cased',
            num_labels=len(self.label_dict),  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                               eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                               )

        self.epochs = 100

        total_steps = len(self.train_dataloader) * self.epochs
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=total_steps)

        self.max_patient = 3

        self.get_class_distribution = data.get_class_distribution

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()

        print(self.device)

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        best_avg_val_loss = 'max'
        unpatient = 0

        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Reset the total loss for this epoch.
            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 10 == 0 and not step == 0:
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()
                '''
                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)
                '''
                l = self.model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

                total_train_loss += l.loss.item()

                l.loss.backward()

                self.optimizer.step()

                self.scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)

            print("")
            print(f"  Average training loss: {avg_train_loss}")

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    l = self.model(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask,
                                                labels=b_labels)
                loss = l.loss
                logits = l.logits
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
            #print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            print(f"  Accuracy: {avg_val_accuracy}")

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)

            #print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print(f"  Validation Loss: {avg_val_loss}")
            print(f'  Validation Best: {best_avg_val_loss}')

            if best_avg_val_loss == 'max' or avg_val_loss < best_avg_val_loss:
                unpatient = 0
                best_avg_val_loss = avg_val_loss
                self.save_model(self.model, self.label_dict, self.conf.PATH_TO_SAVED_MODEL + '/saved_model.pt', epoch_i + 1, avg_val_loss,
                           self.max_len)
            else:
                unpatient += 1

            print(f'  Unpatient: {unpatient}')

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy
                }
            )

            if unpatient >= self.max_patient:
                print('Training complit')
                break

        print("")
        print("Training complete!")

        # Prediction on test set

        print('')
        print('-----------------------------------')
        print('Prediction test')

        # ========================================
        #               Testing
        # ========================================

        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in self.test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')

        total_test_accuracy = self.flat_accuracy(logits, label_ids)
        print(f'total_test_accuracy: {total_test_accuracy}')


    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def save_model(self, model, label_dict, path, epoch, val_loss, max_len):

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'labels_dict': label_dict,
                    'val_acc': val_loss,
                    'epoch': epoch,
                    'max_len': max_len
                    }, path)
        print('  Model saved')

