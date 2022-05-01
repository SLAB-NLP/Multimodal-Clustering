##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from metrics.sensitivity_specificity_metrics.sensitivity_specificity_metric import SensitivitySpecificityMetric
from utils.text_utils import noun_tags


class NounIdentificationMetric(SensitivitySpecificityMetric):
    """ This metric uses the model to predict if a word is a noun (by asking
    if it is associated with a cluster). It then compares the prediction to the
    ground-truth (extracted from a pretrained pos tagger) and reports the
    results. """

    def __init__(self, text_model, nlp):
        super(NounIdentificationMetric, self).__init__(None, text_model)
        self.nlp = nlp
        self.noun_count = 0
        self.non_noun_count = 0

    def prepare_ground_truth(self, text_inputs):
        gt_res = []
        batch_size = len(text_inputs)
        for sample_ind in range(batch_size):
            gt_res.append([])
            doc = self.nlp(' '.join(text_inputs[sample_ind]))
            for token in doc:
                is_noun_gt = token.tag_ in noun_tags
                if is_noun_gt:
                    gt_res[-1].append(1)
                    self.noun_count += 1
                else:
                    gt_res[-1].append(0)
                    self.non_noun_count += 1

        return gt_res

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predictions = self.text_model.predict_cluster_associated_words(text_inputs)
        text_gt = self.prepare_ground_truth(text_inputs)

        batch_size = len(text_inputs)
        for sentence_ind in range(batch_size):
            sentence_len = len(text_inputs[sentence_ind])
            for i in range(sentence_len):
                is_noun_prediction = (predictions[sentence_ind][i] == 1)
                is_noun_gt = (text_gt[sentence_ind][i] == 1)
                if is_noun_prediction and is_noun_gt:
                    self.tp += 1
                elif is_noun_prediction and (not is_noun_gt):
                    self.fp += 1
                elif (not is_noun_prediction) and is_noun_gt:
                    self.fn += 1
                else:
                    self.tn += 1

    def calc_majority_baseline_results(self):
        # Majority baseline means we always predict noun or always predict not-noun
        if self.noun_count > self.non_noun_count:
            # Better to always predict noun
            accuracy = self.noun_count / (self.noun_count + self.non_noun_count)
        else:
            # Better to always predict non-noun
            accuracy = self.non_noun_count / (self.noun_count + self.non_noun_count)

        return accuracy

    def report(self):
        majority_basline_accuracy = self.calc_majority_baseline_results()
        return self.report_with_name() + \
            ', majority baseline accuracy: ' + self.precision_str % majority_basline_accuracy

    def get_name(self):
        return 'Noun prediction'
