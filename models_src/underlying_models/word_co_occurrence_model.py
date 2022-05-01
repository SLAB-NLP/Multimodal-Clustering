##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import numpy as np
from sklearn.cluster import KMeans
from dataset_builders.concreteness_dataset import ConcretenessDatasetBuilder


class WordCoOccurrenceModel:
    """ This model builds words representations based on co-occurrence counts.
        The training set consists of a list of sentences, and in each sentence, each pair of words are considered
        co-occurring.
        We create a co-occurrence matrix, normalize it, and then each row represents a word.
        The building of representations consists of two stages: first, we go over all the training set to build the
        vocabulary, and then we go over it again and build the co-occurrence matrix.
    """

    def __init__(self):
        self.word_to_ind = {}
        self.ind_to_word = []
        self.cur_ind = 0

        self.co_occurrence_matrix = None

    """ Document a word in the vocabulary. Should be called in the vocab building stage. """

    def document_word(self, word):
        if word not in self.word_to_ind:
            self.word_to_ind[word] = self.cur_ind
            self.ind_to_word.append(word)
            self.cur_ind += 1

    """ Document an occurrence of a word in the training set. Should be called during the co-occurrence matrix building
        stage. """

    def document_word_occurrence(self, word):
        word_ind = self.word_to_ind[word]

        self.co_occurrence_matrix[word_ind, word_ind] += 1

    """ Document a co-occurrence of two words in the training set. Should be called during the co-occurrence matrix
        building stage. """

    def document_co_occurrence(self, word1, word2):
        word1_ind = self.word_to_ind[word1]
        word2_ind = self.word_to_ind[word2]

        self.co_occurrence_matrix[word1_ind, word2_ind] += 1
        self.co_occurrence_matrix[word2_ind, word1_ind] += 1

    """ Create the co-occurrence matrix given the collected vocabulary. """

    def create_co_occurrence_matrix(self):
        word_num = len(self.word_to_ind)
        self.co_occurrence_matrix = np.zeros((word_num, word_num))

    """ Predict the concreteness of all the words in the intersection of the vocabulary and the concreteness dataset.
        Concreteness is predicted by taking the 20 most concrete common words in the vocabulary as concrete
        representatives, the 20 least concrete common words in the vocabulary as abstract representative, and for each
        word w, calculating the sum of cosine similarity of w with the concrete representatives minus the sum of cosine
        similarity of w with the abstract representatives.
    """

    def predict_word_concreteness(self, token_count):
        concreteness_dataset = ConcretenessDatasetBuilder(1).build_dataset()
        norm_co_oc_mat = (self.co_occurrence_matrix.transpose() /
                          np.linalg.norm(self.co_occurrence_matrix, axis=1)).transpose()

        # Find concrete and abstract representatives
        common_words_concreteness = [x for x in concreteness_dataset.items()
                                     if x[0] in token_count and token_count[x[0]] > 10]
        common_words_concreteness.sort(key=lambda x: x[1])
        repr_word_per_type_num = 20
        concrete_representatives = [x[0] for x in common_words_concreteness[-repr_word_per_type_num:]]
        abstract_representatives = [x[0] for x in common_words_concreteness[:repr_word_per_type_num]]

        # Create a matrix of the representations of concrete and abstract representatives
        concrete_repr_inds = [self.word_to_ind[w] for w in concrete_representatives]
        concrete_mat = self.co_occurrence_matrix[concrete_repr_inds]
        abstract_repr_inds = [self.word_to_ind[w] for w in abstract_representatives]
        abstract_mat = self.co_occurrence_matrix[abstract_repr_inds]

        # Calculate difference of sum of cosine similarity with concrete and abstract representatives, for each word
        concreteness_mat = np.concatenate([concrete_mat, (-1)*abstract_mat])
        norm_conc_mat = (concreteness_mat.transpose() / np.linalg.norm(concreteness_mat, axis=1)).transpose()

        sim_with_repr_words = np.matmul(norm_co_oc_mat, norm_conc_mat.transpose())
        concreteness_prediction = np.sum(sim_with_repr_words, axis=1)

        return {
            self.ind_to_word[i]: concreteness_prediction[i]
            for i in range(len(self.ind_to_word))
        }

    """ Categorize the words according to the vector representations, using KMeans. """

    def categorize_words(self, word_list, cluster_num):
        all_embeddings_mat = (self.co_occurrence_matrix/np.linalg.norm(self.co_occurrence_matrix, axis=0)).transpose()
        word_indices = [self.word_to_ind[x] for x in word_list]
        embedding_mat = all_embeddings_mat[word_indices]
        kmeans = KMeans(n_clusters=cluster_num).fit(embedding_mat)
        cluster_list = list(kmeans.labels_)
        return {
            word_list[i]: cluster_list[i]
            for i in range(len(word_list))
        }
