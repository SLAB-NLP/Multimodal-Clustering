##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from utils.general_utils import log_print, init_entry_point

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder

# Executors
from executors.clip_positive_threshold_finder import CLIPPositiveThresholdFinder

""" This entry point This class's purpose is to find the best positive-threshold on the MSCOCO training set for the CLIP
model, described in the paper "Learning Transferable Visual Models From Natural Language Supervision" By Radford et al.
For more info, see comments in the file executors.clip_positive_threshold_finder.py. """


def main_find_clip_positive_threshold(write_to_log):
    function_name = 'main_find_clip_positive_threshold'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)

    training_set_config = DatasetConfig(1, include_gt_classes=True)
    training_set, gt_classes_file_path, _ = dataset_builder.build_dataset(training_set_config)
    class_mapping = dataset_builder.get_class_mapping()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Finding best positive threshold...')
    finder = CLIPPositiveThresholdFinder(training_set, gt_classes_file_path, class_mapping, 1)
    finder.find_positive_threshold()
    log_print(function_name, 0, 'Finished')
