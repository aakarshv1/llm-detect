import sys
import gc

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import dill

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

text1 = "Climate change stands as one of the most pressing challenges of our time, with far-reaching consequences that affect ecosystems, societies, and economies around the globe. It is characterized by a significant alteration in Earth's climate patterns, primarily attributed to human activities such as burning fossil fuels, deforestation, and industrial processes. This essay explores the causes and impacts of climate change, highlights the importance of addressing this crisis, and advocates for collective global action to mitigate its effects."
text2 = "Language diversity preserves and promotes distinct cultural worldviews that are essential to innovation. Many linguists subscribe to linguistic relativity, the notion that the language we speak impacts how we think and view the world. In American English we use “thank you” several times a day without batting an eye. But in languages like Hindi, the phrase used to express gratitude, dhanyavad, is so formal that it’s almost never used—using it in everyday situations, like if someone held the door open for you, would probably be interpreted as sarcastic. India is a collectivist society, meaning pleasantries that are common in America aren’t used there because it’s expected that your family and friends will do what they can to help you. In America, where society is more individualistic, helping another person is considered a personal sacrifice—one that should be verbally acknowledged. Using one language instead of the other changes a user’s perception of society, something my parents experienced firsthand when they shifted from a predominantly Hindi-speaking culture to a predominantly English-speaking culture. In that sense, using different languages can cause us to think in fundamentally different ways. And it’s this resulting cultural and intellectual diversity that’s instrumental to progress. Just take a cursory glance at recent Nobel Prize winners: the most cutting edge-discoveries are being made by multilingual teams, and it’s because they offer diverse perspectives that monolingual teams simply can’t.	"


def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text


def classify(text_input):
    # train = pd.read_csv("./model/train_v2_drcat_02.csv", sep=",")
    # y_train = train["label"].values

    with open("./model/tokenizer2.sav", "rb") as file:
        tokenizer = pickle.load(file)

    tokenized_texts_test = []

    tokenized_texts_test.append(tokenizer.tokenize(text_input))

    # tokenized_texts_train = []

    # for text in tqdm(train['text'].tolist()):
    #     tokenized_texts_train.append(tokenizer.tokenize(text))

    with open("./model/vectorizer2.sav", "rb") as file:
        vectorizer = dill.load(file)

    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()

    with open("./model/finalized_model2.sav", "rb") as file:
        ensemble_bytePair = pickle.load(file)

    final_preds_bytePair = ensemble_bytePair.predict_proba(tf_test)[:, 1]

    return f"Probability of being AI-generated: {final_preds_bytePair[0]* 100: .2f}%"


if __name__ == "__main__":
    print(classify(text_input=text2))