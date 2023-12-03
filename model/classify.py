import sys
import gc

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from tokenizers import SentencePieceBPETokenizer

LOWERCASE = False
VOCAB_SIZE = 30522

text_input = "The quick brown fox jumped over the lazy dog."

# Creating Byte-Pair Encoding tokenizer

raw_tokenizer = SentencePieceBPETokenizer()
# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

"""This is for a large test set"""
# Creating huggingface dataset object
# dataset = Dataset.from_pandas(test[['text']])

# def train_corp_iter():
#     """
#     A generator function for iterating over a dataset in chunks.
#     """    
#     for i in range(0, len(dataset), 1000):
#         yield dataset[i : i + 1000]["text"]

# Training from iterator REMEMBER it's training on test set...
# raw_tokenizer.train_from_iterator(train_corp_iter())

"""For single input"""
#raw_tokenizer.train_from_iterator(text_input) #This should be pretrained 

#Load Tokenizer
with open('raw_tokenizer.pkl', 'rb') as file:
    raw_tokenizer = pickle.load(file)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

tokenized_input = tokenizer.tokenize(text_input)
print(tokenized_input)

def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text


vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None#, strip_accents='unicode'
                            )

#load pre-trained vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)



vectorizer.fit(tokenized_input)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)    

# loaded_model = pickle.load(open('./finalized_model.sav', 'rb'))


# try_predict = loaded_model.predict_proba(tf_test)[:,1]