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
from utils.visual_utils import get_resized_gt_bboxes


class CompareToGTBBoxMetric(SensitivitySpecificityMetric):
    """ This is a base class from which the bbox prediction and heatmap metrics will inherit. """

    def __init__(self, visual_model):
        super(CompareToGTBBoxMetric, self).__init__(visual_model, None)

    """ Make a prediction best on the last inference (the inference is assumed to have been done in the Evaluator class
        that instantiated this metric)."""

    @abc.abstractmethod
    def predict(self):
        return

    """ Match predictions to ground-truth boxes according to the specific metric's definition of matching. """

    @abc.abstractmethod
    def match_pred_to_bbox(self, sample_predicted, sample_gt_bboxes):
        return

    """ This function is given the prediction to matching bbox dictionary. Each prediction may match multiple bboxes,
        each bbox may match multiple predictions. For the sake of evaluation, we need to choose not more than one bbox
        per prediction and vice versa. We call this "the final mapping".
        We can have multiple prediction-bbox final mappings. We use a heuristic to find the best one:
        First, we map predictions/bboxes with only one match. Only then we map others. """

    def find_best_pred_bbox_final_mapping(self, pred_to_matching_bbox):
        final_mapping = {}
        mapped_preds = {}
        mapped_bboxes = {}
        # First search for predictions with a single matching bbox
        for pred_ind, bbox_index_list in pred_to_matching_bbox.items():
            if len(bbox_index_list) == 1:
                self.try_mapping_pred_to_bbox(final_mapping, pred_ind, bbox_index_list[0],
                                              mapped_preds, mapped_bboxes)

        # Next, search for bboxes with a single matching prediction
        # - Start by building the reversed dictionary (bbox -> matching pred)
        bbox_to_matching_pred = self.build_bbox_to_matching_pred(pred_to_matching_bbox,
                                                                 mapped_preds, mapped_bboxes)

        # - Find bboxes with a single match
        for bbox_ind, pred_index_list in bbox_to_matching_pred.items():
            if len(pred_index_list) == 1:
                self.try_mapping_pred_to_bbox(final_mapping, pred_index_list[0], bbox_ind,
                                              mapped_preds, mapped_bboxes)

        # Finally, match unmatched
        # - Update the bbox -> matching pred dict (need to do it since we want to ignore preds and bboxes that were
        #   already mapped
        bbox_to_matching_pred = self.build_bbox_to_matching_pred(pred_to_matching_bbox,
                                                            mapped_preds, mapped_bboxes)
        # - Go over the list and try mapping every bbox to each of matching preds, until one succeeds
        for bbox_ind, pred_index_list in bbox_to_matching_pred.items():
            while len(pred_index_list) > 0:
                self.try_mapping_pred_to_bbox(final_mapping, pred_index_list[0], bbox_ind,
                                              mapped_preds, mapped_bboxes)
                pred_index_list = pred_index_list[1:]

        return final_mapping

    """ Check if the provided prediction and gt bbox are not already matched. If they don't, match them. """

    @staticmethod
    def try_mapping_pred_to_bbox(final_mapping, pred_ind, bbox_ind, mapped_preds, mapped_bboxes):
        if pred_ind not in mapped_preds and bbox_ind not in mapped_bboxes:
            final_mapping[pred_ind] = bbox_ind
            mapped_preds[pred_ind] = True
            mapped_bboxes[bbox_ind] = True

    """ Given the pred to matching bbox dict, build the reversed dict, ignoring bboxes and preds that were already
        mapped. """

    @staticmethod
    def build_bbox_to_matching_pred(pred_to_matching_bbox, mapped_preds, mapped_bboxes):
        bbox_to_matching_pred = {}

        for pred_ind, bbox_index_list in pred_to_matching_bbox.items():
            if pred_ind not in mapped_preds:
                for bbox_ind in bbox_index_list:
                    if bbox_ind not in mapped_bboxes:
                        if bbox_ind not in bbox_to_matching_pred:
                            bbox_to_matching_pred[bbox_ind] = []
                        bbox_to_matching_pred[bbox_ind].append(pred_ind)

        return bbox_to_matching_pred

    def document(self, orig_image_sizes, predicted_list, gt_bboxes):
        batch_size = len(gt_bboxes)
        for sample_ind in range(batch_size):
            sample_gt_bboxes = gt_bboxes[sample_ind]
            gt_bbox_num = len(sample_gt_bboxes)
            sample_gt_bboxes = get_resized_gt_bboxes(sample_gt_bboxes, orig_image_sizes[sample_ind])
            sample_predicted = predicted_list[sample_ind]
            predicted_num = len(sample_predicted)

            # Create prediction to matching bbox dictionary
            pred_to_matching_bbox = self.match_pred_to_bbox(sample_predicted, sample_gt_bboxes)

            # Determine final prediction -> bbox mapping
            final_mapping = self.find_best_pred_bbox_final_mapping(pred_to_matching_bbox)

            ''' True/false positive/negative: every mapping we created is considered a true positive by definition
            (because every mapped pred->bbox were matched before). All the predictions/bbox for which no mapping was
            found are considered false positive/negative. '''
            tp = len(final_mapping)
            fp = predicted_num - tp
            fn = gt_bbox_num - tp

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predictions = self.predict()
        gt_bboxes = visual_metadata['gt_bboxes']
        orig_image_sizes = visual_metadata['orig_image_size']
        self.document(orig_image_sizes, predictions, gt_bboxes)

    def report(self):
        return self.report_with_name()

    @staticmethod
    def is_image_only():
        return True
