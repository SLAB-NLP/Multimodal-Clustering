##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.demonstrators.demonstrator import Demonstrator
from models_src.wrappers.visual_model_wrapper import VisualModelWrapper
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper
import os
from utils.general_utils import visual_dir, text_dir


class HeatmapDemonstrator(Demonstrator):

    def __init__(self, model_name, dataset, class_mapping,
                 num_of_items_to_demonstrate, indent):
        super(HeatmapDemonstrator, self).__init__(dataset, num_of_items_to_demonstrate, indent)

        visual_model_dir = os.path.join(self.models_dir, visual_dir)
        text_model_dir = os.path.join(self.models_dir, text_dir)
        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.text_model.eval()

        cluster_to_gt_class_ind = self.text_model.create_cluster_to_gt_class_mapping(class_mapping)
        cluster_to_gt_class_str = {x: [class_mapping[i] for i in cluster_to_gt_class_ind[x]]
                                   for x in cluster_to_gt_class_ind.keys()}

        self.cluster_to_gt_class_str = cluster_to_gt_class_str

    def demonstrate_item(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image']
        self.visual_model.demonstrate_heatmap(image_tensor, self.cluster_to_gt_class_str)
