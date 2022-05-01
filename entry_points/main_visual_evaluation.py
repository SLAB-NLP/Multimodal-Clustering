##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
from utils.general_utils import log_print, init_entry_point, models_dir, visual_dir, text_dir

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.evaluators.visual_evaluators.common_visual_evaluator import CommonVisualEvaluator

""" This entry point evaluate the model on the most common visual metrics. """


def main_visual_evaluation(write_to_log, model_name):
    function_name = 'main_visual_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, slice_str, _ = create_dataset_builder(dataset_name)

    test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True, include_gt_bboxes=True)
    test_set, gt_classes_file_path, gt_bboxes_file_path = dataset_builder.build_dataset(test_set_config)
    class_mapping = dataset_builder.get_class_mapping()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    visual_model_dir = os.path.join(models_dir, visual_dir)
    text_model_dir = os.path.join(models_dir, text_dir)
    evaluator = CommonVisualEvaluator(visual_model_dir, text_model_dir, model_name, test_set,
                                      gt_classes_file_path, gt_bboxes_file_path, class_mapping, 1)
    evaluator.evaluate()
    log_print(function_name, 0, 'Finished testing')
