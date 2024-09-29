import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re
import seaborn as sns
import pickle
import streamlit as st

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn_crfsuite.utils import flatten
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from random import shuffle
from time import time
from sklearn_crfsuite.metrics import flat_accuracy_score, flat_precision_score, flat_recall_score, flat_f1_score, flat_fbeta_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict

with open('classifier.pkl','rb') as file:
    model = pickle.load(file)

nltk.download('brown')
nltk.download('universal_tagset')
dataset_crf = list(nltk.corpus.brown.tagged_sents(tagset = "universal"))
dataset_hmm = list(nltk.corpus.brown.tagged_sents(tagset = "universal"))

for sent in dataset_hmm:
    temp = ("^", "START")
    sent.insert(0, temp)

tags_temp = set()
tokens_temp = set()
for sent in dataset_hmm:
    for ele in sent:
        tokens_temp.add(ele[0])
        tags_temp.add(ele[1])
tags = list(tags_temp)
tokens = list(tokens_temp)
n_tags = len(tags)
n_tokens = len(tokens)

def training(dataset):
    transition_freq = defaultdict(int)
    emission_freq = defaultdict(int)
    tag_freq = defaultdict(int)

    for sent in dataset:
        prev_tag = "START"
        for tup in sent:
            transition_freq[(prev_tag, tup[1])] += 1
            emission_freq[(tup[1], tup[0])] += 1
            tag_freq[tup[1]] += 1
            prev_tag = tup[1]
    
    a = 1e-8
    transition_mat  = [[0 for _ in range(n_tags)] for _ in range(n_tags)]
    emission_mat = [[0 for _ in range(n_tokens)] for _ in range(n_tags)]
    
    for i in range(n_tags):
        for j in range(n_tags):
            transition_mat[i][j] = (transition_freq[(tags[i], tags[j])] + a) / (tag_freq[tags[i]] + a*n_tags)
        for j in range(n_tokens):
            emission_mat[i][j] = (emission_freq[(tags[i], tokens[j])] + a) / (tag_freq[tags[i]] + a*n_tags)

    return transition_mat, emission_mat

def preprocess_sent(sent):
    words = sent.split()
    exception_punc = ['\'', '-']
    proc_words = []
    for w in words:
        if (w == " "):
            continue
        else:
            if (w != "I"):
                w = w.lower()
            s = 0
            n = len(w)
            for i in range(n):
                if (w[i].isalnum()):
                    continue
                elif (w[i] not in exception_punc):
                    proc_words.append(w[s:i])
                    proc_words.append(w[i])
                    s = i+1
            if (s < n):
                proc_words.append(w[s:])
    if (proc_words[-1].isalnum()):
        proc_words.append(".")
    proc_words[0] = proc_words[0].capitalize()
    for ele in proc_words:
        if (ele == ''):
            proc_words.remove(ele)
    return (proc_words)

def Viterbi_decoder(t_mat, e_mat, sent):
    em_min = np.array(e_mat).min()
    words = preprocess_sent(sent)
    #print(words)
    n = len(words)
    prob_mat = [[0 for _ in range(n_tags)] for _ in range(n)]
    tag_mat = [[0 for _ in range(n_tags)] for _ in range(n)]

    ind = tags.index("START")
    try:
        w_ind = tokens.index(words[0])
    except ValueError:
        w_ind = -1
    for i in range(n_tags):
        if (w_ind == -1):
            prob_mat[0][i] = t_mat[ind][i] * em_min
        else:
            prob_mat[0][i] = t_mat[ind][i] * e_mat[i][tokens.index(words[0])]

    for s in range(1, n):
        prev_token = words[s-1]
        curr_token = words[s]
        
        try:
            w_curr_ind = tokens.index(curr_token)
        except ValueError:
            w_curr_ind = -1

        for t in range(n_tags):
            p = 0
            temp_tag = 0
            for u in range(n_tags):
                temp_p = prob_mat[s-1][u] * t_mat[u][t]
                if temp_p > p:
                    p = temp_p
                    temp_tag = u
            if (w_curr_ind == -1):
                prob_mat[s][t] = p * em_min
            else:
                prob_mat[s][t] = p * e_mat[t][tokens.index(curr_token)]
            tag_mat[s][t] = temp_tag

    pred = []
    # Get the last tag 
    pmax = 0
    last_tag = 0
    for i in range(n_tags):
        if prob_mat[-1][i] > pmax:
            pmax = prob_mat[-1][i]
            last_tag = i
    pred.append(tags[last_tag])

    t = last_tag
    for i in range(n-1, 0, -1):
        t = int(tag_mat[i][t])
        pred.append(tags[t])

    pred.reverse()
    return (words, pred)

tm, em = training(dataset_hmm)

def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word': word,
        'word_length': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word': word1,
            '-1:word_length': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word': word1,
            '+1:word_length': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]



st.title("POS Tagging with HMM and CRF")
st.markdown("Enter a sentence below and click the button to get the POS tags.")

# Input field for the sentence
sent = st.text_input("Enter the sentence:")

# Button to trigger the processing
if st.button("Get POS Tags"):
    if sent:
        # Preprocess the sentence
        words = preprocess_sent(sent)

        # Timing and prediction with CRF
        crf_st = time()
        output_crf = model.predict([sent2features(words)])
        crf_end = time()

        # Timing and prediction with HMM
        output_hmm = Viterbi_decoder(tm, em, sent)
        hmm_end = time()

        # Prepare the results
        d = {
            "Words": output_hmm[0],
            "HMM POS Tags": output_hmm[1],
            "CRF POS Tags": flatten(output_crf)
        }
        df = pd.DataFrame(d)

        # Display results
        st.subheader("Results:")
        st.dataframe(df)

        # Display timing results
        st.write(f"**Time taken by HMM:** {hmm_end - crf_end:.4f} seconds")
        st.write(f"**Time taken by CRF:** {crf_end - crf_st:.4f} seconds")
    else:
        st.warning("Please enter a sentence before clicking the button.")
