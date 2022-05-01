##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from metrics.sensitivity_specificity_metrics.sensitivity_specificity_metric import SensitivitySpecificityMetric


class VisualClassificationMetric(SensitivitySpecificityMetric):
    """ Base class for image multi-label classification metrics. """

    def __init__(self, visual_model, class_num):
        super(VisualClassificationMetric, self).__init__(visual_model, None)
        self.class_num = class_num

    def evaluate_classification(self, predicted_classes, gt_classes):
        batch_size = len(predicted_classes)
        for sample_ind in range(batch_size):
            sample_predicted = predicted_classes[sample_ind]
            sample_gt = gt_classes[sample_ind]
            cur_tp, cur_fp, cur_fn, cur_tn = self.calculate_sens_spec_metrics(sample_predicted, sample_gt)

            self.tp += cur_tp
            self.fp += cur_fp
            self.fn += cur_fn
            self.tn += cur_tn

    """ Calculate the number of true positive, false positive, false negatives, true negatives given a list of predicted
        classes and the list of ground-truth classes.
        For each class, we count it's occurrences in both lists, and use it to find these metrics.
    """

    def calculate_sens_spec_metrics(self, predicted_classes, gt_classes):
        unique_predicted_classes = set(predicted_classes)
        unique_gt_classes = set(gt_classes)

        predicted_num = len(unique_predicted_classes)
        gt_num = len(unique_gt_classes)

        tp_count = len(unique_predicted_classes.intersection(unique_gt_classes))
        fp_count = predicted_num - tp_count
        fn_count = gt_num - tp_count

        non_predicted_num = self.class_num - predicted_num
        tn_count = non_predicted_num - fn_count

        return tp_count, fp_count, fn_count, tn_count

    @staticmethod
    def count_class_instances(inst_list):
        class_to_count = {}
        for class_inst in inst_list:
            if class_inst not in class_to_count:
                class_to_count[class_inst] = 0
            class_to_count[class_inst] += 1
        return class_to_count

    @abc.abstractmethod
    def document(self, predicted_classes, gt_classes):
        return

    @staticmethod
    def is_image_only():
        return True
