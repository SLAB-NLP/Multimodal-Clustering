##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import numpy as np
from metrics.metric import Metric


class ConcretenessPredictionMetric(Metric):
    """ This metric uses a concreteness dataset: a human-annotated dataset
        that gives every word a number between 1 and 5, representing how
        concrete is this word. We compare it to our prediction.
    """

    def __init__(self, text_model, concreteness_dataset, token_count):
        super(ConcretenessPredictionMetric, self).__init__(None, text_model)
        self.concreteness_dataset = concreteness_dataset
        self.prediction_absolute_error_sum = 0
        self.tested_words_count = 0

        ''' We want to evaluate concreteness prediction on different sets of words: words that appeared more than once
        in the training set, words that appeared more than 5 times in the training set, etc. '''
        self.token_count = token_count
        self.min_count_vals = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        self.min_count_gt_lists = []
        self.min_count_pred_lists = []
        for _ in self.min_count_vals:
            self.min_count_gt_lists.append([])
            self.min_count_pred_lists.append([])

    """ Go over the entire validation set (words that occurred in our training set, i.e. are in token_count, intersected
        with words that occur in the external concreteness dataset), predict concreteness, and compare to ground-truth.
    """

    def traverse_validation_set(self):
        for token, count in self.token_count.items():
            if token not in self.concreteness_dataset:
                continue

            self.tested_words_count += 1
            gt_concreteness = self.concreteness_dataset[token]
            concreteness_prediction = self.text_model.estimate_word_concreteness([[token]])[0][0]

            prediction_absolute_error = abs(gt_concreteness - concreteness_prediction)
            self.prediction_absolute_error_sum += prediction_absolute_error

            for i in range(len(self.min_count_vals)):
                val = self.min_count_vals[i]
                if count > val:
                    self.min_count_gt_lists[i].append(gt_concreteness)
                    self.min_count_pred_lists[i].append(concreteness_prediction)

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return  # Nothing to do here, this metric uses an external dataset

    def calc_results(self):
        self.results = {}
        self.traverse_validation_set()

        concreteness_prediction_mae = 0
        if self.tested_words_count > 0:
            concreteness_prediction_mae = \
                self.prediction_absolute_error_sum / self.tested_words_count
        self.results['concreteness prediction mae'] = concreteness_prediction_mae

        for i in range(len(self.min_count_vals)):
            gt_and_predictions = np.array([self.min_count_gt_lists[i], self.min_count_pred_lists[i]])
            pred_pearson_corr = np.corrcoef(gt_and_predictions)[0, 1]
            self.results['prediction correlation over ' + str(self.min_count_vals[i])] = \
                (pred_pearson_corr, len(self.min_count_gt_lists[i]))

    def report(self):
        if self.results is None:
            self.calc_results()

        res = ''
        res += 'concreteness prediction mean absolute error: ' + \
               self.precision_str % self.results['concreteness prediction mae'] + ', '

        res += 'pearson correlation by token count: '
        for val in self.min_count_vals:
            if val != self.min_count_vals[0]:
                res += ', '
            res += str(val) + ': '
            cur_result = self.results['prediction correlation over ' + str(val)]
            res += str((self.precision_str % cur_result[0], cur_result[1]))

        return res

    @staticmethod
    def uses_external_dataset():
        return True
