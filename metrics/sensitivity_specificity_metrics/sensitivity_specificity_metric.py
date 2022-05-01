##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from metrics.metric import Metric


class SensitivitySpecificityMetric(Metric):
    """ This class represents metrics with the concepts of true/false positive/negatives. """

    def __init__(self, visual_model, text_model):
        super(SensitivitySpecificityMetric, self).__init__(visual_model, text_model)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def calc_results(self):
        if self.results is None:
            self.results = {}
        name = self.get_name()

        if self.tp + self.fp == 0:
            precision = 0
        else:
            precision = self.tp / (self.tp + self.fp)
        self.results[name + ' precision'] = precision

        if self.tp + self.fn == 0:
            recall = 0
        else:
            recall = self.tp / (self.tp + self.fn)
        self.results[name + ' recall'] = recall

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        self.results[name + ' F1'] = f1

    def report_with_name(self):
        if self.results is None:
            self.calc_results()

        name = self.get_name()
        res = name + ': '
        res += 'tp ' + str(self.tp)
        res += ', fp ' + str(self.fp)
        res += ', fn ' + str(self.fn)
        res += ', tn ' + str(self.tn)

        res += ', precision: ' + self.precision_str % self.results[name + ' precision']
        res += ', recall: ' + self.precision_str % self.results[name + ' recall']
        res += ', F1: ' + self.precision_str % self.results[name + ' F1']

        return res

    @abc.abstractmethod
    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        return

    @abc.abstractmethod
    def report(self):
        return

    @abc.abstractmethod
    def get_name(self):
        return
