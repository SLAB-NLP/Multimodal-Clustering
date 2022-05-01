##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
import os
import random
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import multiple_word_string
from utils.visual_utils import get_image_shape_from_id
import torch
from datasets_src.img_captions_dataset.single_img_captions_dataset import SingleImageCaptionDataset
from datasets_src.img_dataset import ImageDataset
from dataset_builders.classification_dataset_builder import ClassificationDatasetBuilder


class ImageCaptionDatasetBuilder(ClassificationDatasetBuilder):
    """ This class is the base class for all external image-caption datasets builders. """

    def __init__(self, name, indent):
        super(ImageCaptionDatasetBuilder, self).__init__(indent)
        self.name = name

        self.slices = ['train', 'test']
        self.file_paths = {}
        for slice_str in self.slices:
            self.file_paths[slice_str] = self.get_filepaths_for_slice(slice_str)

        self.val_data_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_val_data')

    # Implemented methods

    """ Get the paths of files containing the data, for a specific slice of the dataset. """

    def get_filepaths_for_slice(self, slice_str):
        return {
            'captions': os.path.join(self.cached_dataset_files_dir, self.name + '_captions_' + slice_str),
            'gt_classes': os.path.join(self.cached_dataset_files_dir, self.name + '_gt_classes_' + slice_str),
            'gt_bboxes': os.path.join(self.cached_dataset_files_dir, self.name + '_gt_bboxes_' + slice_str)
        }

    def generate_caption_data(self, slice_str):
        return generate_dataset(self.file_paths[slice_str]['captions'], self.generate_caption_data_internal, slice_str)

    """ Separate a part of the wanted slice (specified using orig_slice_str) as validation data.
        ratio_from_orig_slice specifies the side of the validation data. If force_create==True, we'll override existing
        validation data.
    """

    def generate_val_data_split(self, orig_slice_str, ratio_from_orig_slice, force_create=False):
        if force_create and os.path.isfile(self.val_data_file_path):
            os.remove(self.val_data_file_path)
        return generate_dataset(self.val_data_file_path, self.generate_val_data_split_internal,
                                orig_slice_str, ratio_from_orig_slice)

    def generate_val_data_split_internal(self, orig_slice_str, ratio_from_orig_slice):
        caption_data = self.generate_caption_data(orig_slice_str)
        orig_slice_size = len(caption_data)
        val_data_size = int(ratio_from_orig_slice*orig_slice_size)

        val_indices = random.sample(range(orig_slice_size), val_data_size)
        return val_indices

    # Abstract methods

    """ Generate a list of dictionaries that contain image id and a corresponding caption. For example:
        [
            {'image_id': 123, 'caption': 'A large dog'},
            {'image_id': 456, 'caption': 'A white airplane'},
            ...
        ]
    """

    @abc.abstractmethod
    def generate_caption_data_internal(self, slice_str):
        return

    """ Generate a mapping from image id to the list of class indices in this image. For example:
        {123: [1,2,6], 456: [1,3], ...}
    """

    @abc.abstractmethod
    def generate_gt_classes_data(self, slice_str):
        return

    """ Generate a mapping from image id to the list of bounding boxes in this image. Each bounding box is 4 integers:
    (left x-axis edge, upper y-axis edge, right x-axis edge, lower y-axis edge). For example:
        {
            123: [[10, 5, 90, 200], [40, 180, 70, 190]],
            456: [[100, 100, 110, 150]],
            ...
        }
    """

    @abc.abstractmethod
    def generate_gt_bboxes_data(self, slice_str):
        return

    """ Generates both the ground-truth class and bounding boxes at the same time.
        This may be more convenient as in some of the datasets, both are stored in the same place.
    """

    @abc.abstractmethod
    def generate_gt_classes_bboxes_data(self, slice_str):
        return

    """ Get the path to the file of a given image id, in a given slice of the dataset. """

    @abc.abstractmethod
    def get_image_path(self, image_id, slice_str):
        return

    @abc.abstractmethod
    def get_class_mapping(self):
        return

    # Current class specific functionality

    """ We want to filter images that are:
        - Grayscale
        - Contain multiple-words-named classes
        - Without bbox or classes ground-truth data
        This function returns a list of image ids of images we want to filter.
    """

    def find_unwanted_images(self, slice_str):
        caption_dataset = self.generate_caption_data(slice_str)
        image_ids_by_caption_dataset = list(set([x['image_id'] for x in caption_dataset]))

        img_classes_dataset, img_bboxes_dataset = self.generate_gt_classes_bboxes_data(slice_str)
        class_mapping = self.get_class_mapping()

        multiple_word_classes = [x for x in class_mapping.keys() if multiple_word_string(class_mapping[x])]
        self.log_print('Multiple word classes: ' + str([class_mapping[x] for x in multiple_word_classes]))

        self.unwanted_images_info = {
            'img_classes_dataset': img_classes_dataset,
            'img_bboxes_dataset': img_bboxes_dataset,
            'multiple_word_classes': multiple_word_classes,
            'grayscale_count': 0,
            'multiple_word_classes_count': 0,
            'no_class_or_bbox_data_count': 0,
            'unwanted_image_ids': [],
            'slice_str': slice_str
        }

        self.increment_indent()
        for_loop_with_reports(image_ids_by_caption_dataset, len(image_ids_by_caption_dataset),
                              10000, self.is_unwanted_image, self.unwanted_images_progress_report)
        self.decrement_indent()

        self.log_print('Out of ' + str(len(img_classes_dataset)) + ' images:')
        self.log_print('Found ' + str(self.unwanted_images_info['no_class_or_bbox_data_count']) +
                       ' without class or bbox data')
        self.log_print('Found ' + str(self.unwanted_images_info['grayscale_count']) + ' grayscale images')
        self.log_print('Found ' + str(self.unwanted_images_info['multiple_word_classes_count']) +
                       ' multiple word classes')

        return self.unwanted_images_info['unwanted_image_ids']

    """ This function checks if current image should be filtered, and if so, adds it to the unwanted image list. """

    def is_unwanted_image(self, index, item, print_info):
        image_id = item

        # No class or bbox data
        if (self.unwanted_images_info['img_classes_dataset'] is not None and
            image_id not in self.unwanted_images_info['img_classes_dataset']) or \
                (self.unwanted_images_info['img_bboxes_dataset'] is not None and
                 image_id not in self.unwanted_images_info['img_bboxes_dataset']):
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['no_class_or_bbox_data_count'] += 1
            return

        gt_classes = self.unwanted_images_info['img_classes_dataset'][image_id]

        # Grayscale
        image_shape = get_image_shape_from_id(image_id, self.get_image_path, self.unwanted_images_info['slice_str'])
        if len(image_shape) == 2:
            # Grayscale images only has 2 dims
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['grayscale_count'] += 1
            return

        # Multiple word classes
        if len(set(gt_classes).intersection(self.unwanted_images_info['multiple_word_classes'])) > 0:
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['multiple_word_classes_count'] += 1
            return

    def unwanted_images_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting image ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))

    """ We want to filter images that are:
        - Grayscale
        - Contain multiple-words-named classes
        - Without bbox or classes ground-truth data
    """

    def filter_unwanted_images(self, slice_str):
        unwanted_image_ids = self.find_unwanted_images(slice_str)

        caption_dataset = self.generate_caption_data(slice_str)
        img_classes_dataset, img_bboxes_dataset = self.generate_gt_classes_bboxes_data(slice_str)

        new_caption_dataset = [x for x in caption_dataset if x['image_id'] not in unwanted_image_ids]
        new_img_classes_dataset = {x: img_classes_dataset[x] for x in img_classes_dataset.keys()
                                   if x not in unwanted_image_ids}
        new_img_bboxes_dataset = {x: img_bboxes_dataset[x] for x in img_bboxes_dataset.keys()
                                  if x not in unwanted_image_ids}

        torch.save(new_caption_dataset, self.file_paths[slice_str]['captions'])
        torch.save(new_img_classes_dataset, self.file_paths[slice_str]['gt_classes'])
        torch.save(new_img_bboxes_dataset, self.file_paths[slice_str]['gt_bboxes'])

    """ Return a dataset object containing only images, and ignoring the captions. """

    def build_image_only_dataset(self, config):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        file_paths = self.file_paths[config.slice_str]
        if config.include_gt_classes:
            self.generate_gt_classes_data(config.slice_str)
            gt_classes_file_path = file_paths['gt_classes']
        else:
            gt_classes_file_path = None
        if config.include_gt_bboxes:
            self.generate_gt_bboxes_data(config.slice_str)
            gt_bboxes_file_path = file_paths['gt_bboxes']
        else:
            gt_bboxes_file_path = None

        self.generate_caption_data(config.slice_str)

        return ImageDataset(file_paths['captions'], self.get_image_path, config), \
            gt_classes_file_path, \
            gt_bboxes_file_path

    # Inherited methods

    def build_dataset(self, config=None):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        file_paths = self.file_paths[config.slice_str]

        self.generate_caption_data(config.slice_str)
        if config.include_gt_classes:
            self.generate_gt_classes_data(config.slice_str)
            gt_classes_file_path = file_paths['gt_classes']
        else:
            gt_classes_file_path = None
        if config.include_gt_bboxes:
            self.generate_gt_bboxes_data(config.slice_str)
            gt_bboxes_file_path = file_paths['gt_bboxes']
        else:
            gt_bboxes_file_path = None

        class_mapping = self.get_class_mapping()

        # Validation data: if we need only validation data (or anything but the validation data), create a list of
        # relevant indices
        val_data_indices = None
        if config.exclude_val_data or config.only_val_data:
            val_data_indices = self.generate_val_data_split('train', 0.2)

        return \
            SingleImageCaptionDataset(file_paths['captions'],
                                      gt_classes_file_path,
                                      class_mapping,
                                      self.get_image_path,
                                      config,
                                      val_data_indices), \
            gt_classes_file_path, \
            gt_bboxes_file_path
