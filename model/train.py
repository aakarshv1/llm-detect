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
from sklearn.ensemble import RandomForestClassifier
# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  

LOWERCASE = False
VOCAB_SIZE = 30522

text_input = "Climate change stands as one of the most pressing challenges of our time, with far-reaching consequences that affect ecosystems, societies, and economies around the globe. It is characterized by a significant alteration in Earth's climate patterns, primarily attributed to human activities such as burning fossil fuels, deforestation, and industrial processes. This essay explores the causes and impacts of climate change, highlights the importance of addressing this crisis, and advocates for collective global action to mitigate its effects."
train = pd.read_csv("train_v2_drcat_02.csv", sep=',')
y_train = train['label'].values

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
# Creating huggingface dataset object
dataset = Dataset.from_pandas(train[['text']])
def train_corp_iter(): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

filename = 'tokenizer2.sav'
pickle.dump(tokenizer, open(filename, 'wb'))


tokenized_texts_test = []

tokenized_texts_test.append(tokenizer.tokenize(text_input))

tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text
    
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode'
                            )

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)

filename = 'vectorizer2.sav'
pickle.dump(vectorizer, open(filename, 'wb'))

tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()

clf = MultinomialNB(alpha=0.02)
# clf2 = MultinomialNB(alpha=0.01)
sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
p6={'n_iter': 1000,'verbose': -1,'learning_rate': 0.005689066836106983, 'colsample_bytree': 0.8915976762048253, 'colsample_bynode': 0.5942203285139224, 'lambda_l1': 7.6277555139102864, 'lambda_l2': 6.6591278779517808, 'min_data_in_leaf' : 156, 'max_depth': 11, 'max_bin': 813}
lgb=LGBMClassifier(**p6)

ensemble_bytePair = VotingClassifier(estimators=[('mnb',clf),('sgd', sgd_model),('lgb',lgb),('rf', rf_model)],
                            weights=[0.25,0.25,0.25,0.25], voting='soft', n_jobs=-1)

ensemble_bytePair.fit(tf_train, y_train)

gc.collect()

model_save = ensemble_bytePair
filename = 'finalized_model2.sav'
pickle.dump(model_save, open(filename, 'wb'))

final_preds_bytePair = ensemble_bytePair.predict_proba(tf_test)[:,1]

print(final_preds_bytePair)