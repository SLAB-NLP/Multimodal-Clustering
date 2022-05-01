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

# Models
from models_src.wrappers.visual_classifier_using_text import CLIPVisualClassifier


class ClipClassificationEvaluator(VisualEvaluator):
    """ This class evaluates the CLIP model, described in the paper "Learning Transferable Visual Models From Natural
        Language Supervision" By Radford et al. on the multi-label visual classification task.
    """

    def __init__(self, test_set, gt_classes_file_path, gt_bboxes_file_path, class_mapping, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

        self.model = CLIPVisualClassifier(0.19031381607055664, class_mapping, self.indent+1)
        class_num = len([x for x in class_mapping.items() if ' ' not in x[1]])

        self.metrics = [
            VisualClassNameClassificationMetric(
                self.model,
                class_num
            )
        ]

    """ Run inference on input, using the evaluated model. """

    def infer(self, visual_input, visual_metadata):
        self.model.inference(visual_input)
