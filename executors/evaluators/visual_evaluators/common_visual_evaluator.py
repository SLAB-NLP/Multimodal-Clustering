##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.evaluators.visual_evaluators.visual_evaluator import VisualEvaluator

# Metrics
from metrics.sensitivity_specificity_metrics.visual_classification_metrics.visual_class_name_classification_metric \
    import VisualClassNameClassificationMetric
from metrics.sensitivity_specificity_metrics.visual_classification_metrics.visual_cluster_classification_metric \
    import VisualClusterClassificationMetric
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.bbox_prediction_metric \
    import BBoxPredictionMetric
from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.heatmap_metric import HeatmapMetric

# Models
from models_src.wrappers.visual_model_wrapper import VisualModelWrapper
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper
from models_src.wrappers.visual_classifier_using_text import ClusterVisualClassifier


class CommonVisualEvaluator(VisualEvaluator):
    """ This evaluator is the most commonly used for visual tasks. """

    def __init__(self, visual_model_dir, text_model_dir, model_name, test_set,
                 gt_classes_file_path, gt_bboxes_file_path, class_mapping, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        self.class_mapping = class_mapping

        # Models
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.text_model.eval()

        class_num = len([x for x in class_mapping.items() if ' ' not in x[1]])

        self.metrics = [
            VisualClassNameClassificationMetric(
                ClusterVisualClassifier(self.visual_model, self.text_model, class_mapping, self.indent+1),
                class_num
            ),
            BBoxPredictionMetric(self.visual_model),
            HeatmapMetric(self.visual_model),
            VisualClusterClassificationMetric(self.visual_model, class_num, 'co_occur'),
            VisualClusterClassificationMetric(self.visual_model, class_num, 'iou')
        ]

    """ Run inference on input, using the evaluated model. """

    def infer(self, visual_input, visual_metadata):
        self.visual_model.inference(visual_input)
