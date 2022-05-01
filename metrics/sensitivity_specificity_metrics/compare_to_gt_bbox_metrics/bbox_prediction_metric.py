##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.compare_to_gt_bbox_metric \
    import CompareToGTBBoxMetric
from utils.visual_utils import calc_ious, predict_bboxes_with_activation_maps


class BBoxPredictionMetric(CompareToGTBBoxMetric):
    """ This metric predicts bounding boxes, compares the prediction to the
    ground-truth bounding boxes (using intersection-over-union), and reports
    the results. """

    def __init__(self, visual_model):
        super(BBoxPredictionMetric, self).__init__(visual_model)

    def match_pred_to_bbox(self, sample_predicted_bboxes, sample_gt_bboxes):
        """ A predicted bounding box is considered matching a ground-truth bounding box if the Intersection-over-Union
            (IoU) of the two is larger than 0.5 (a commonly used threshold).
        """
        gt_bbox_num = len(sample_gt_bboxes)
        sample_gt_bboxes = torch.stack([torch.tensor(x) for x in sample_gt_bboxes])
        if len(sample_predicted_bboxes) > 0:
            sample_predicted_bboxes = torch.stack([torch.tensor(x) for x in sample_predicted_bboxes])
            ious = calc_ious(sample_gt_bboxes, sample_predicted_bboxes)
        pred_to_matching_gt = {}
        for predicted_bbox_ind in range(len(sample_predicted_bboxes)):
            pred_to_matching_gt[predicted_bbox_ind] = []
            for gt_bbox_ind in range(gt_bbox_num):
                if ious[gt_bbox_ind, predicted_bbox_ind] >= 0.5:
                    pred_to_matching_gt[predicted_bbox_ind].append(gt_bbox_ind)

        return pred_to_matching_gt

    def predict(self):
        return self.visual_model.predict_bboxes()

    """ Document the metric values without predicting, given the predicted activation maps. """

    def document_with_loaded_results(self, orig_image_sizes, activation_maps, gt_bboxes):
        predicted_bboxes = predict_bboxes_with_activation_maps(activation_maps)
        self.document(orig_image_sizes, predicted_bboxes, gt_bboxes)

    def get_name(self):
        return 'bbox prediction'
