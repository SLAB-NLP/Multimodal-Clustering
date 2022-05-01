##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import spacy

""" This file contains functions and definitions related to text used across all the project. """

nlp = spacy.load('en_core_web_sm')
tokenizer = nlp.tokenizer


noun_tags = [
    'NN',
    'NNP',
    'NNS',
    'NNPS',
    'PRP'
]

verb_tags = [
    'VBZ',
    'VB',
    'VBP',
    'VBG',
    'VBN',
    'VBD'
]

adjective_tags = [
    'JJ',
    'JJR',
    'JJS'
]


def is_noun(pos_tag):
    return pos_tag in noun_tags


def preprocess_token(token):
    token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!"))
    token = token.lower()

    return token


def prepare_data(captions, lemmatize=False):
    """ Tokenize and clean a list of input captions. """
    token_lists = []
    for caption in captions:
        if lemmatize:
            doc = nlp(caption)
            token_list = [str(x.lemma_) for x in doc]
        else:
            token_list = [str(x) for x in list(tokenizer(caption.lower()))]
        token_lists.append(token_list)
    return token_lists


def multiple_word_string(my_str):
    """ Check if the input string has multiple words. """
    return len(my_str.split()) > 1
