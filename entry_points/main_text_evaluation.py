##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
from utils.general_utils import log_print, init_entry_point, models_dir, text_dir

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.evaluators.common_text_evaluator import CommonTextEvaluator

""" This entry point evaluate the model on the most common text metrics. """


def main_text_evaluation(write_to_log, model_name):
    function_name = 'main_text_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)

    training_set_config = DatasetConfig(1)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    token_count = training_set.get_token_count()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    text_model_dir = os.path.join(models_dir, text_dir)
    evaluator = CommonTextEvaluator(text_model_dir, model_name, token_count, 1)
    evaluator.evaluate()
    log_print(function_name, 0, 'Finished testing')
