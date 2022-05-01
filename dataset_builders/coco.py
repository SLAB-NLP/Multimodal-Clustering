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
import json
from dataset_builders.img_caption_dataset_builder import ImageCaptionDatasetBuilder


class Coco(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the MSCOCO dataset, described in the paper
        'Microsoft COCO: Common Objects in Context' by Lin et al.
        Something weird about COCO: They published 2 splits: train, val, test, but they didn't provide labels for the
        test split. So we're going to use the val set as a test set, and split the training set to training and val.
    """

    def __init__(self, root_dir_path, indent):
        super(Coco, self).__init__('coco', indent)
        self.root_dir_path = root_dir_path

        self.train_val_annotations_dir = 'train_val_annotations2014'

        self.train_bboxes_filepath_suffix = 'instances_train2014.json'
        self.train_bboxes_filepath = os.path.join(root_dir_path, self.train_val_annotations_dir,
                                                  self.train_bboxes_filepath_suffix)
        self.val_bboxes_filepath_suffix = 'instances_val2014.json'
        self.val_bboxes_filepath = os.path.join(root_dir_path, self.train_val_annotations_dir,
                                                self.val_bboxes_filepath_suffix)

        self.train_captions_filepath_suffix = os.path.join(self.train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_filepath = os.path.join(root_dir_path, self.train_captions_filepath_suffix)
        self.val_captions_filepath_suffix = os.path.join(self.train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_filepath = os.path.join(root_dir_path, self.val_captions_filepath_suffix)

        self.train_images_dirpath = os.path.join(root_dir_path, 'train2014')
        self.val_images_dirpath = os.path.join(root_dir_path, 'val2014')

    def generate_caption_data_internal(self, slice_str):
        if slice_str == 'train':
            external_caption_filepath = self.train_captions_filepath
        elif slice_str == 'test':
            external_caption_filepath = self.val_captions_filepath
        caption_fp = open(external_caption_filepath, 'r')
        caption_data = json.load(caption_fp)
        return caption_data['annotations']

    def generate_gt_classes_data(self, slice_str):
        self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_bboxes_data(self, slice_str):
        self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_classes_bboxes_data(self, slice_str):
        gt_classes_filepath = self.file_paths[slice_str]['gt_classes']
        gt_bboxes_filepath = self.file_paths[slice_str]['gt_bboxes']

        if os.path.exists(gt_classes_filepath):
            return torch.load(gt_classes_filepath), torch.load(gt_bboxes_filepath)
        else:
            if slice_str == 'train':
                external_bboxes_filepath = self.train_bboxes_filepath
            elif slice_str == 'test':
                external_bboxes_filepath = self.val_bboxes_filepath
            bboxes_fp = open(external_bboxes_filepath, 'r')
            bboxes_data = json.load(bboxes_fp)

            category_id_to_class_id = {bboxes_data[u'categories'][x][u'id']: x for x in
                                       range(len(bboxes_data[u'categories']))}

            img_classes_dataset = {}
            img_bboxes_dataset = {}
            # Go over all the object annotations
            for bbox_annotation in bboxes_data[u'annotations']:
                image_id = bbox_annotation[u'image_id']
                if image_id not in img_bboxes_dataset:
                    img_classes_dataset[image_id] = []
                    img_bboxes_dataset[image_id] = []

                # First, extract the bounding box
                bbox = bbox_annotation[u'bbox']
                xmin = int(bbox[0])
                xmax = int(bbox[0] + bbox[2])
                ymin = int(bbox[1])
                ymax = int(bbox[1] + bbox[3])
                trnsltd_bbox = [xmin, ymin, xmax, ymax]

                # Next, extract the ground-truth class of this object
                category_id = bbox_annotation[u'category_id']
                class_id = category_id_to_class_id[category_id]

                img_classes_dataset[image_id].append(class_id)
                img_bboxes_dataset[image_id].append(trnsltd_bbox)

            torch.save(img_classes_dataset, gt_classes_filepath)
            torch.save(img_bboxes_dataset, gt_bboxes_filepath)

            return img_classes_dataset, img_bboxes_dataset

    def get_class_mapping(self):
        bbox_fp = open(self.train_bboxes_filepath, 'r')
        bbox_data = json.load(bbox_fp)

        category_id_to_class_id = {bbox_data[u'categories'][x][u'id']: x for x in range(len(bbox_data[u'categories']))}
        category_id_to_name = {x[u'id']: x[u'name'] for x in bbox_data[u'categories']}
        class_mapping = {category_id_to_class_id[x]: category_id_to_name[x] for x in category_id_to_class_id.keys()}

        return class_mapping

    def get_image_path(self, image_id, slice_str):
        if slice_str == 'test':
            coco_slice_name = 'val'
        else:
            coco_slice_name = 'train'
        image_filename = 'COCO_' + coco_slice_name + '2014_000000' + '{0:06d}'.format(image_id) + '.jpg'
        if slice_str == 'train':
            images_dirpath = self.train_images_dirpath
        elif slice_str == 'test':
            images_dirpath = self.val_images_dirpath
        image_path = os.path.join(images_dirpath, image_filename)

        return image_path
