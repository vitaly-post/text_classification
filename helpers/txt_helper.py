from . import regexp_compilation as rc
import nltk
import string
from transformers import BertTokenizer
#
#PRETRAINED_BERT_DIR = r'../deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v1'


def make_lower(text):
    return text.lower()

def tockenize_documents(all_documents):
    new_all_documents = []
    for line in all_documents:
        new_all_documents.append(nltk.word_tokenize(line))

    return new_all_documents


def clear_doc_for_ner(doc):
    #pure_text = rc.remove_set_9.sub(r' ', doc)
    pure_text = rc.remove_set_10.sub(r' ', doc)
    pure_text = pure_text.replace('\n', ' ')
    pure_text = pure_text.replace('\r', ' ')
    pure_text = pure_text.replace('\t', ' ')
    pure_text = pure_text.replace('\xa0', ' ')

    pure_text = rc.multiple_spaces.sub(r' ', pure_text)
    pure_text = pure_text.strip()

    return pure_text


def get_tokens_count(message, tokenizer):
    # bert tokenizer
    #tokenizer = BertTokenizer.from_pretrained(ner_conf['PATH_TO_PRETRAINED_BERT_MODEL'] + r'/vocab.txt')
    tokens = tokenizer.tokenize(message)

    return len(tokens)


def split_message(message, ratio='half'):
    if ratio == 'half':
        lst_message = message.split(' ')
        split_num = len(lst_message) / 2
        lst_first_message = ''
        lst_second_message = ''
        for i, word in enumerate(lst_message):
            if i < split_num:
                lst_first_message += word + ' '
            else:
                lst_second_message += word + ' '

        return lst_first_message.strip(), lst_second_message.strip()

    return message


def is_list_of_strings(lst):
        return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)


def gluing_B_I(text_lst, tags_lst):
    #if len(lst_tags_text) == 0:
    #    return []

    name = None
    lst_names = []
    #res = list(zip(*lst_tags_text))

    #if lst_tags_text[0][1].split('-')[0] == 'I':
    #    lst_tags_text[0] = lst_tags_text[0][0], 'B-' + lst_tags_text[0][1].split('-')[1]

    for i, tags in enumerate(tags_lst):
        if tags != 'O':
            prefix, tag = tags.split('-')
            if prefix == 'B':
                if name is not None:
                    lst_names.append(name)

                name = text_lst[i], tags

            elif prefix == 'I':
                if name is None:
                    name = text_lst[i], tags
                    lst_names.append(name)

                else:
                    name = name[0] + ' ' + text_lst[i], tags

    if name is not None:
        lst_names.append(name)
    lst_names_pure_tag = []

    if len(lst_names) > 0:
        for item in lst_names:
            new_item = (item[0], item[1].split('-')[1])
            lst_names_pure_tag.append(new_item)

    return lst_names_pure_tag


def clear_text_for_classification(text):
    pure_text = text.replace('\n', ' ')
    pure_text = rc.remove_set_6.sub(r' ', pure_text)
    pure_text = rc.underscores.sub(r' ', pure_text)

    pure_text = rc.multiple_spaces.sub(r' ', pure_text)
    pure_text = pure_text.strip()

    return pure_text.lower()

