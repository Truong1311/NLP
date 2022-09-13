import os
import re
import torch
import torchtext
import nltk
from nltk.corpus import stopwords
from ntlk.stem import PorterStemmer
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, LabelField

STOPWORDS = stopwords.words('english')

def preprocessing_text(texts):
    cleaned_texts = []
    for text in texts:
        # lower text
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # remove multiple space
        text = re.sub(r' +', ' ', text)
        # remove url
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # remove stopwords and stemming words
        ps = PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split() if word not in STOPWORDS])
        cleaned_texts.append(text)
    return cleaned_texts

def load_file(path, tokenize, batch_size, fix_length, max_size, device):
    '''
    Input: 
    - path: path to folder of data
    - tokenize: 
    - batch_size: size of batch for training 
    - fix_length: max number of words in a sentence
    - max_size: max number of words in vocabulary
    Output:
    vocabylary size after create from  train data, train_iter, valid_iter, test_iter, Text , Label 
    '''
    Text = Field(sequential= True,
             tokenize= tokenize,
             preprocessing= preprocessing_text,
             batch_first= True,
             include_lengths= True,
             fix_length = fix_length)
    Label = LabelField(dtype = torch.float, batch_first = True)
    train_fields = [('id', None), ('keyword', None), ('location', None), ('text', Text), ('target', Label)]
    train_dataset = TabularDataset(path = path + 'train.csv', format = 'csv', fields = train_fields, skip_header = True)

    Text.build_vocab(train_dataset, max_size = max_size)
    Label.build_vocab(train_dataset)
    vocab_size = len(Text.vocab)
    # split train and valid set
    train_dataset, valid_dataset = train_dataset.split(split_ratio = 0.8)
    # tranform to iterator
    train_iter, valid_iter = BucketIterator.splits((train_dataset, valid_dataset),
                                                    batch_size = batch_size,
                                                    sort_key= lambda x: len(x.text),
                                                    device= device, repeat = False, shuffle= True)
    # test dataset
    test_fields = [('id', None), ('keyword', None), ('location', None), ('text', Text)]
    test_dataset = TabularDataset(path = path + 'test.csv', format = 'csv', fields = test_fields, skip_header= True)
    test_iter = BucketIterator(test_dataset,
                                    batch_size = batch_size,
                                    sort_key = lambda x: len(x.text), 
                                    device = device)
    return vocab_size, train_iter, valid_iter, test_iter, Text, Label

