##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
import torch.utils.data as data
from utils.general_utils import for_loop_with_reports
from utils.multi_label_threshold_finder import find_best_threshold
from executors.executor import Executor
from models_src.wrappers.clip_wrapper import CLIPWrapper


class CLIPPositiveThresholdFinder(Executor):
    """ This class's purpose is to find the best positive-threshold for the CLIP model on the MSCOCO training set.
        CLIP (Radford et al. 2021) was designed for a single-label classification setting. It contains a visual encoder
        and a text encoder, both encoding the input to a joint embedding space. Given the names of all ground-truth
        classes, the model is given a prompt for each class (e.g., for the ground-truth class DOG the model is given the
        sentence "a photo of a dog"). The model encodes all prompts and a given input image, and classifies the image to
        the class with which it has the largest cosine similarity.
        In multi-classification class, there might be more than one class in an image. Therefore, we need to set a
        threshold, where classes with cosine similarity higher than the threshold will be considered poritive, and all
        other classes will be considered negative. The purpose of this class is to find the threshold that maximizes F1
        score on MSCOCO training set.
    """

    def __init__(self, training_set, gt_classes_file_path, class_mapping, indent):
        super().__init__(indent)

        self.training_set = training_set
        self.clip_wrapper = CLIPWrapper(class_mapping, indent + 1)
        self.clip_wrapper.initialize()
        self.gt_classes_data = torch.load(gt_classes_file_path)

    """ The entry point of this class. """

    def find_positive_threshold(self):
        self.log_print('Collecting similarity-ground-truth list...')
        self.create_sim_gt_list()
        self.log_print('Finished collecting similarity-ground-truth list')

        self.log_print('Searching for the best threshold...')
        best_threshold = find_best_threshold(self.sim_gt_list, self.indent + 1)
        self.log_print('Found the best threshold: ' + str(best_threshold))

    def create_sim_gt_list(self):
        self.sim_gt_list = []

        dataloader = data.DataLoader(self.training_set, batch_size=50)

        checkpoint_len = 100
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.collect_sim_gt_from_batch, self.progress_report)
        self.decrement_indent()

    def collect_sim_gt_from_batch(self, index, sampled_batch, print_info):
        with torch.no_grad():
            visual_inputs = sampled_batch['image'].to(self.device)
            image_features = self.clip_wrapper.encode_image(visual_inputs)
            similarity_mat = self.clip_wrapper.calculate_similarity(image_features)

            image_ids = [x.item() for x in sampled_batch['image_id']]
            gt_classes = [self.gt_classes_data[x] for x in image_ids]

            class_num = self.clip_wrapper.get_class_num()
            batch_size = similarity_mat.shape[0]
            for mat_class_ind in range(class_num):
                gt_class_ind = self.clip_wrapper.mat_ind_to_class_ind[mat_class_ind]
                for sample_ind in range(batch_size):
                    cur_sim = similarity_mat[sample_ind, mat_class_ind]
                    is_gt_in_sample = gt_class_ind in gt_classes[sample_ind]
                    self.sim_gt_list.append((cur_sim.item(), is_gt_in_sample))
