##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from loggable_object import LoggableObject
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import POS_LIST
import numpy as np
from sklearn import svm
from dataset_builders.fast_text_dataset_builder import FastTextDatasetBuilder


class ConcretenessSupervisedModel(LoggableObject):
    """ This class is a supervised model for concreteness prediction.
        It's an SVM regression model that uses part-of-speech tags, common suffixes and pretrained embeddings as
        features.
        The model was presented in the paper "Predicting  word  concreteness  and  imagery" by Jean Charbonnier and
        Christian Wartena.
    """

    def __init__(self, use_pos_tags, use_suffix, use_embeddings, suffix_num, fast_text_dim, indent):
        super(ConcretenessSupervisedModel, self).__init__(indent)

        self.suffix_num = suffix_num
        self.fast_text_dim = fast_text_dim

        # Each word is converted to a vector using its input features.
        # Find the dimension of this vector
        self.word_vec_dim = 0
        if use_pos_tags:
            self.word_vec_dim += len(POS_LIST)
        if use_suffix:
            self.word_vec_dim += self.suffix_num
        if use_embeddings:
            self.word_vec_dim += self.fast_text_dim

        self.use_pos_tags = use_pos_tags
        self.use_suffix = use_suffix
        self.use_embeddings = use_embeddings

    """ Get the part-of-speech vector for a given word.
        The part-of-speech vector is a vector of the normalized counts of this word part-of-speech tags in the WordNet
        dataset.
    """

    @staticmethod
    def get_pos_vector(word):
        res_vec = [len(wn.synsets(word, pos=pos_tag)) for pos_tag in POS_LIST]
        synset_num = sum(res_vec)
        if synset_num > 0:
            res_vec = [x/synset_num for x in res_vec]
        else:
            # This is an unknown words, return 0 for all pos tags
            res_vec = [0]*len(POS_LIST)

        return np.array(res_vec)

    """ Find the most common suffixes in out training set. """

    def find_common_suffixes(self, training_set):
        # First go over the entire training set and create a suffix->count mapping
        suffix_count = {}
        suffix_max_len = 4
        for word in training_set.keys():
            for suffix_len in range(1, suffix_max_len + 1):
                cur_suffix = word[-suffix_len:]
                if cur_suffix not in suffix_count:
                    suffix_count[cur_suffix] = 0
                suffix_count[cur_suffix] += 1

        # Now, find the self.suffix_num most common suffixes
        all_suffixes = list(suffix_count.items())
        all_suffixes.sort(key=lambda x: x[1], reverse=True)
        self.common_suffixes = [x[0] for x in all_suffixes[:self.suffix_num]]

    """ Get the common suffixes vector of a given word.
        The common suffixes vector is a vector of size self.suffix_num, where the ith entry is 1 if the current word has
        the ith common suffix, and 0 otherwise. 
    """

    def get_suffix_vector(self, word):
        res_vec = [
            1 if word.endswith(self.common_suffixes[i]) else 0
            for i in range(self.suffix_num)
        ]
        return np.array(res_vec)

    """ Get the pretrained fast-text embedding vector for the input word. """

    def get_embedding_vector(self, word):
        if word in self.fast_text:
            return self.fast_text[word]
        else:
            # Fast-text isn't familiar with this word
            return np.zeros(self.fast_text_dim)

    """ Get the vector representation of the input word, using the model features. """

    def get_word_vector(self, word):
        word_vector = np.array([])
        if self.use_pos_tags:
            word_vector = np.concatenate([word_vector, self.get_pos_vector(word)])
        if self.use_suffix:
            word_vector = np.concatenate([word_vector, self.get_suffix_vector(word)])
        if self.use_embeddings:
            word_vector = np.concatenate([word_vector, self.get_embedding_vector(word)])

        return word_vector

    """ Train the SVM regression model on the given training set.
        This function returns a mapping of word->concreteness prediction.
    """

    def train(self, training_set):
        if self.use_suffix:
            self.log_print('Collecting common suffixes...')
            self.find_common_suffixes(training_set)
        if self.use_embeddings:
            self.fast_text = \
                FastTextDatasetBuilder({x: True for x in training_set.keys()}, self.indent + 1).build_dataset()

        X = np.zeros((len(training_set), self.word_vec_dim))
        y = np.zeros(len(training_set))
        word_to_ind = {}
        ind_to_word = []
        self.log_print('Collecting word vectors for all words...')
        for word, concreteness in training_set.items():
            cur_ind = len(ind_to_word)
            word_vector = self.get_word_vector(word)
            X[cur_ind, :] = word_vector
            y[cur_ind] = concreteness

            word_to_ind[word] = cur_ind
            ind_to_word.append(word)

        self.log_print('Training SVM...')
        regr = svm.SVR()
        regr.fit(X, y)

        self.log_print('Predicting using trained model...')
        predictions = regr.predict(X)

        return {
            ind_to_word[i]: predictions[i] for i in range(len(ind_to_word))
        }
