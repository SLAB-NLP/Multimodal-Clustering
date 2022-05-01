##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from metrics.metric import Metric


class WordAssociationMetric(Metric):
    """ This metric estimates how associated are words in a given clustering solution.
    """

    def __init__(self, text_model, word_list, assoc_strength_dataset):
        super(WordAssociationMetric, self).__init__(None, text_model)
        self.word_list = word_list
        self.assoc_strength_dataset = assoc_strength_dataset

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        # This metric is not related to the test set, all the measurements are done later
        return

    def report(self):
        if self.results is None:
            self.calc_results()

        res = 'Word association: '
        res += 'Mean association strength: ' + \
               self.precision_str % self.results['mas'] + ', '
        res += 'number of found pairs: ' + \
               self.precision_str % self.results['found'] + ', '
        res += 'number of not found pairs: ' + \
               self.precision_str % self.results['not_found']

        return res

    def calc_results(self):
        self.results = {}

        mas, found, not_found = self.evaluate_clusters()
        self.results['mas'] = mas
        self.results['found'] = found
        self.results['not_found'] = not_found

    """ Evaluate the association strength of pairs in our induced clusters. """

    def evaluate_clusters(self):
        word_to_cluster = {word: self.text_model.predict_cluster_for_word(word)[0] for word in self.word_list}
        cluster_num = max(word_to_cluster.values()) + 1
        cluster_to_word_list = {i: [x for x in self.word_list if word_to_cluster[x] == i] for i in range(cluster_num)}
        cluster_pair_lists = [
            [x for outer in [[(z[i], z[j]) for j in range(i + 1, len(z))] for i in range(len(z))] for x in outer] for z
            in cluster_to_word_list.values()]
        all_pair_lists = [x for outer in cluster_pair_lists for x in outer]

        strength_sum = 0
        found = 0
        not_found = 0
        for x in all_pair_lists:
            word1 = sorted([x[0], x[1]])[0]
            word2 = sorted([x[0], x[1]])[1]
            if word1 in self.assoc_strength_dataset and word2 in self.assoc_strength_dataset[word1]:
                strength_sum += self.assoc_strength_dataset[word1][word2]
                found += 1
            else:
                not_found += 1
        return strength_sum / found, found, not_found

    @staticmethod
    def uses_external_dataset():
        return True
