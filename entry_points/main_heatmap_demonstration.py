##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

# General
import os
from utils.general_utils import log_print, init_entry_point

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Executors
from executors.demonstrators.heatmap_demonstrator import HeatmapDemonstrator


""" This entry point demonstrates the heatmaps predicted by our model using Class Activation Mapping. """


def main_heatmap_demonstration(write_to_log, model_name):
    function_name = 'main_heatmap_demonstration'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, slice_str, _ = create_dataset_builder(dataset_name)

    test_set_config = DatasetConfig(1, slice_str=slice_str, include_gt_classes=True, include_gt_bboxes=True)
    test_set, _, _ = dataset_builder.build_dataset(test_set_config)
    class_mapping = dataset_builder.get_class_mapping()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Demonstrating heatmaps...')
    demonstrator = HeatmapDemonstrator(model_name, test_set, class_mapping, 3, 1)
    demonstrator.demonstrate()
