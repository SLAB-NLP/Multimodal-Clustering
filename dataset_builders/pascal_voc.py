##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import os
from xml.dom import minidom
import torch
from dataset_builders.classification_dataset_builder import ClassificationDatasetBuilder
from datasets_src.img_dataset import ImageDataset
from utils.general_utils import generate_dataset


class PascalVOC(ClassificationDatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(PascalVOC, self).__init__(indent)
        self.root_dir_path = root_dir_path
        self.name = 'pascal_voc'

        self.image_dir = os.path.join(self.root_dir_path, 'JPEGImages')
        self.annotation_dir = os.path.join(self.root_dir_path, 'Annotations')

        self.image_id_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_image_id')
        self.gt_classes_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_gt_classes')
        self.class_mapping_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_class_mapping')

    @staticmethod
    def file_name_to_image_id(file_name):
        return int(file_name.split('.')[0].replace('_', ''))

    def generate_image_id_list(self):
        return generate_dataset(self.image_id_file_path, self.generate_image_id_list_internal)

    def generate_image_id_list_internal(self):
        image_id_list = [self.file_name_to_image_id(x) for x in os.listdir(self.image_dir)]
        return image_id_list

    def create_image_id_to_gt_class_str_mapping(self):
        image_id_to_gt_classes_str = {}

        for _, _, files in os.walk(self.annotation_dir):
            for filename in files:
                # Extract image file name from current file name
                image_id = self.file_name_to_image_id(filename)

                # Extract ground-truth classes from file
                gt_classes = []

                xml_filepath = os.path.join(self.annotation_dir, filename)
                xml_doc = minidom.parse(xml_filepath)
                for child_node in xml_doc.childNodes[0].childNodes:
                    # The bounding boxes are located inside a node named "object"
                    if child_node.nodeName == u'object':
                        # Go over all of the children of this node: if we find bndbox, this object is a bounding box
                        for inner_child_node in child_node.childNodes:
                            if inner_child_node.nodeName == u'name':
                                class_name = inner_child_node.childNodes[0].data
                                gt_classes.append(class_name)

                image_id_to_gt_classes_str[image_id] = gt_classes

        return image_id_to_gt_classes_str

    def generate_gt_classes_data(self):
        self.generate_class_mapping_and_gt_classes_data()

    def generate_class_mapping_and_gt_classes_data(self):
        if os.path.exists(self.class_mapping_file_path):
            return torch.load(self.class_mapping_file_path), torch.load(self.gt_classes_file_path)
        else:
            image_id_to_gt_classes_str = self.create_image_id_to_gt_class_str_mapping()
            all_classes = list(set([inner for outer in image_id_to_gt_classes_str.values() for inner in outer]))
            class_mapping = {x: all_classes[x] for x in range(len(all_classes))}
            reverse_class_mapping = {all_classes[x]: x for x in range(len(all_classes))}
            image_id_to_gt_classes = {
                x[0]: [reverse_class_mapping[y] for y in x[1]] for x in image_id_to_gt_classes_str.items()
            }

            torch.save(class_mapping, self.class_mapping_file_path)
            torch.save(image_id_to_gt_classes, self.gt_classes_file_path)

            return class_mapping, image_id_to_gt_classes

    def get_class_mapping(self):
        self.generate_class_mapping_and_gt_classes_data()
        return torch.load(self.class_mapping_file_path)

    def get_image_path(self, image_id, slice_str):
        image_filename_prefix = f'{image_id // 1000000:04d}'
        image_filename_suffix = f'{image_id % 1000000:06d}'
        image_filename = image_filename_prefix + '_' + image_filename_suffix + '.jpg'
        image_path = os.path.join(self.image_dir, image_filename)

        return image_path

    def build_dataset(self, config=None):
        self.generate_image_id_list()
        image_id_list = torch.load(self.image_id_file_path)

        if config.include_gt_classes:
            self.generate_gt_classes_data()
            gt_classes_file_path = self.gt_classes_file_path
        else:
            gt_classes_file_path = None

        return ImageDataset(image_id_list, self.get_image_path, config), \
            gt_classes_file_path, \
            None
