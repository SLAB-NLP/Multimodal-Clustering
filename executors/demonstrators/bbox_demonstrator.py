##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
import torch
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from models_src.wrappers.visual_model_wrapper import VisualModelWrapper
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper

from utils.general_utils import visual_dir, text_dir
from utils.visual_utils import get_resized_gt_bboxes, wanted_image_size, unnormalize_trans, \
    predict_bboxes_with_activation_maps

from executors.demonstrators.demonstrator import Demonstrator


class BboxDemonstrator(Demonstrator):

    def __init__(self, model_name, dataset, gt_classes_file_path, gt_bboxes_file_path, class_mapping,
                 num_of_items_to_demonstrate, indent):
        super(BboxDemonstrator, self).__init__(dataset, num_of_items_to_demonstrate, indent)

        visual_model_dir = os.path.join(self.models_dir, visual_dir)
        text_model_dir = os.path.join(self.models_dir, text_dir)

        self.visual_model = VisualModelWrapper(self.device, None, visual_model_dir, model_name, indent + 1)
        self.visual_model.eval()
        self.text_model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.text_model.eval()

        self.gt_classes_data = torch.load(gt_classes_file_path)
        self.gt_bboxes_data = torch.load(gt_bboxes_file_path)
        self.class_mapping = class_mapping
        self.cluster_to_class = self.text_model.create_cluster_to_gt_class_mapping(class_mapping)

    def demonstrate_item(self, index, sampled_batch, print_info):
        image_tensor = sampled_batch['image']
        image_tensor = unnormalize_trans(image_tensor)
        image_id = sampled_batch['image_id'].item()

        activation_maps_with_clusters = self.visual_model.predict_activation_maps_with_clusters(image_tensor)[0]
        predicted_clusters = [x[0] for x in activation_maps_with_clusters]
        activation_maps = [x[1] for x in activation_maps_with_clusters]
        predicted_bboxes = predict_bboxes_with_activation_maps([activation_maps])[0]
        gt_bboxes = get_resized_gt_bboxes(self.gt_bboxes_data[image_id], sampled_batch['orig_image_size'])
        gt_classes = self.gt_classes_data[image_id]

        image_obj = to_pil_image(image_tensor.view(3, wanted_image_size[0], wanted_image_size[1]))
        draw = ImageDraw.Draw(image_obj)

        # First draw ground-truth boxes
        for bbox_ind in range(len(gt_bboxes)):
            bbox = gt_bboxes[bbox_ind]
            gt_class = gt_classes[bbox_ind]
            draw.rectangle(bbox, outline=(255, 0, 0))
            text_loc = (bbox[0], bbox[1])
            draw.text(text_loc, self.class_mapping[gt_class], fill=(255, 0, 0))

        # Next, draw predicted boxes
        for bbox_ind in range(len(predicted_bboxes)):
            bbox = predicted_bboxes[bbox_ind]
            predicted_cluster = predicted_clusters[bbox_ind]
            draw.rectangle(bbox, outline=(0, 255, 0))
            text_loc = (bbox[0], bbox[1])
            predicted_golden_classes = self.cluster_to_class[predicted_cluster]
            draw.text(text_loc, str([self.class_mapping[x] for x in predicted_golden_classes]), fill=(0, 255, 0))

        plt.imshow(image_obj)
        plt.show()
