##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import numpy as np


class WordClusterCountModel:
    """ This class is a simple count model for word clustering.
        The class can be activated in two modes: discriminative or generative.
        The model uses counts of word-cluster co-occurrences to predict a class given a word. """

    def __init__(self, cluster_num, mode):
        self.cluster_num = cluster_num
        if mode == 'discriminative':
            """ In discriminative mode, we calculate the probability of a cluster given a word p(c|w) directly.
                We maintain a word cluster count for each word cluster pair N(w,c), and calculate:
                p(c|w) = N(w,c)/N(w)
            """
            self.word_to_cluster_co_occur = {}
            self.mode = 0  # 0 means discriminative
        elif mode == 'generative':
            """ In generative mode, we calculate the probability of a cluster given a word using Bayes rule:
                p(c|w) = (p(w|c) * p(c)) / p(w)
                p(w|c) is estimated using the word, cluster count N(w,c):
                p(w|c) = N(w,c) / N(c)
                P(c) is set to 1/N where N is the number of clusters.
                p(w) is estimated as the sum of p(c1)*p(w|c1), p(c2)*p(w|c2), ..., p(cN)*p(w|cN).
            """
            self.cluster_to_word_co_occur = []
            for _ in range(cluster_num):
                self.cluster_to_word_co_occur.append({})

            # We'll keep a lazy calculation of p(w|c). The real value is updated using the calculate probs method
            self.cluster_to_word_prob = []
            self.cluster_prob = []

            self.mode = 1  # 1 means generative
        else:
            print('Error: no such mode ' + str(mode))
            assert False

    """ Document a word, cluster observed co-occurrence. """

    def document_co_occurrence(self, word, cluster_ind):
        if self.mode == 0:
            if word not in self.word_to_cluster_co_occur:
                self.word_to_cluster_co_occur[word] = [0] * self.cluster_num
            self.word_to_cluster_co_occur[word][cluster_ind] += 1
        else:
            if word not in self.cluster_to_word_co_occur[cluster_ind]:
                self.cluster_to_word_co_occur[cluster_ind][word] = 0
            self.cluster_to_word_co_occur[cluster_ind][word] += 1

    """ Update the lazy calculation of p(w|c) and p(c). """

    def calculate_probs(self):
        if self.mode == 0:
            # Nothing to do in discriminative mode
            return
        else:
            self.cluster_to_word_prob = []
            for cluster_ind in range(self.cluster_num):
                word_count_dic = self.cluster_to_word_co_occur[cluster_ind]
                cluster_total_occurrence_num = sum(word_count_dic.values())
                self.cluster_to_word_prob.append({
                    x[0]: x[1] / cluster_total_occurrence_num for x in word_count_dic.items()
                })

            # Use uniform class distribution
            self.cluster_prob = [1 / self.cluster_num for _ in range(self.cluster_num)]

    def get_cluster_conditioned_on_word(self, word):
        # First, get p(ci)*p(word|ci) for each cluster ci
        cluster_to_prob = [self.cluster_prob[x]*self.cluster_to_word_prob[x][word]
                           if word in self.cluster_to_word_prob[x]
                           else 0
                           for x in range(self.cluster_num)]

        # Next, estimate p(word) by summing p(ci)*p(word|ci) for each ci
        word_prob_sum = sum(cluster_to_prob)
        if word_prob_sum == 0:
            # print('Never encountered word \'' + word + '\'.')
            return None

        # Finally, estimate p(c|w) for each cluster
        p_cluster_cond_word = [(cluster_to_prob[x]) / word_prob_sum
                               for x in range(self.cluster_num)]

        return p_cluster_cond_word

    """ Return the most probable cluster given a word, and its probability. """

    def predict_cluster(self, word):
        if self.mode == 0:
            if word not in self.word_to_cluster_co_occur:
                # print('Never encountered word \'' + word + '\'.')
                return None

            highest_correlated_cluster = np.argmax(self.word_to_cluster_co_occur[word])

            # Find N(w,c)
            highest_count = self.word_to_cluster_co_occur[word][highest_correlated_cluster]

            # Find N(w)
            overall_count = sum(self.word_to_cluster_co_occur[word])

            # Estimate p(c|w) using N(w,c)/N(w)
            probability = highest_count / overall_count

            most_probable_cluster = highest_correlated_cluster
        else:
            p_class_cond_word = self.get_cluster_conditioned_on_word(word)
            if p_class_cond_word is None:
                # print('Never encountered word \'' + word + '\'.')
                return None
            probability = max(p_class_cond_word)
            most_probable_cluster = np.argmax(p_class_cond_word)

        return most_probable_cluster, probability
