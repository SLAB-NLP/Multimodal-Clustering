##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.evaluators.visual_evaluators.visual_evaluator import VisualEvaluator


class RandomBBoxEvaluator(VisualEvaluator):
    """ This evaluator evaluates metrics related to bounding boxes using random guesses.
        We assume self.model inherit from RandomPredictor.
    """

    def __init__(self, test_set, gt_classes_file_path, gt_bboxes_file_path, indent):
        super().__init__(test_set, gt_classes_file_path, gt_bboxes_file_path, indent)

    def infer(self, visual_input, visual_metadata):
        gt_bboxes = visual_metadata['gt_bboxes']
        self.model.set_prediction_num([len(x) for x in gt_bboxes])
