##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.evaluators.visual_evaluators.random_bbox_evaluators.random_bbox_evaluator import RandomBBoxEvaluator

# Metrics
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.bbox_prediction_metric import BBoxPredictionMetric

# Models
from models_src.wrappers.visual_model_wrapper import BBoxRandomPredictionModelWrapper


class RandomBBoxPredictionEvaluator(RandomBBoxEvaluator):
    """ This evaluator evaluates the bounding-box prediction metric using random guesses. """

    def __init__(self, test_set, gt_classes_file_path, gt_bboxes_file_path, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        # Model
        self.model = BBoxRandomPredictionModelWrapper()

        self.metrics = [BBoxPredictionMetric(self.model)]
