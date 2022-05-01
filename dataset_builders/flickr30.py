##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import os
import torch
from xml.dom import minidom
from utils.general_utils import generate_dataset
from dataset_builders.img_caption_dataset_builder import ImageCaptionDatasetBuilder


class Flickr30(ImageCaptionDatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(Flickr30, self).__init__('flickr30', indent)
        self.root_dir_path = root_dir_path

        tokens_dir_name = 'tokens'
        tokens_file_name = 'results_20130124.token'
        self.tokens_file_path = os.path.join(self.root_dir_path, tokens_dir_name, tokens_file_name)

        bbox_dir_name = os.path.join('annotations', 'Annotations')
        self.bbox_dir_path = os.path.join(self.root_dir_path, bbox_dir_name)

        sentences_dir_name = os.path.join('annotations', 'Sentences')
        self.sentences_dir_path = os.path.join(self.root_dir_path, sentences_dir_name)

        images_dir_name = 'images'
        self.images_dir_path = os.path.join(self.root_dir_path, images_dir_name)

        self.boxes_chains_filename = os.path.join(self.cached_dataset_files_dir, 'flickr30_boxes_chains')
        self.coord_strs = ['xmin', 'ymin', 'xmax', 'ymax']
        self.coord_str_to_ind = {self.coord_strs[x]: x for x in range(len(self.coord_strs))}
        self.chains_and_classes_filename = os.path.join(self.cached_dataset_files_dir, 'flickr30_chains_and_classes')

    def generate_caption_data_internal(self, slice_str):
        if slice_str == 'train':
            # No training for flickr30k
            return []
        elif slice_str == 'test':
            fp = open(self.tokens_file_path, encoding='utf-8')
            image_id_captions_pairs = []
            for line in fp:
                split_line = line.strip().split('#')
                img_file_name = split_line[0]
                image_id = self.image_filename_to_id(img_file_name)
                caption = split_line[1].split('\t')[1]  # The first token is caption number

                image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

            return image_id_captions_pairs

    @staticmethod
    def image_filename_to_id(image_filename):
        return int(image_filename.split('.')[0])

    def generate_gt_classes_data(self, slice_str):
        if slice_str == 'val':
            self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_bboxes_data(self, slice_str):
        if slice_str == 'val':
            self.generate_gt_classes_bboxes_data(slice_str)

    def generate_gt_classes_bboxes_data(self, slice_str):
        gt_classes_filepath = self.file_paths[slice_str]['gt_classes']
        gt_bboxes_filepath = self.file_paths[slice_str]['gt_bboxes']

        if os.path.exists(gt_classes_filepath):
            return torch.load(gt_classes_filepath), torch.load(gt_bboxes_filepath)
        else:
            boxes_chains, chain_list = self.extract_boxes_and_chains()
            chain_to_class_ind, class_ind_to_str = self.get_chain_to_class_mapping(chain_list)
            img_classes_dataset = {y: [chain_to_class_ind[x[1]] for x in boxes_chains[y]]
                                   for y in boxes_chains.keys()}
            img_bboxes_dataset = {y: [x[0] for x in boxes_chains[y]] for y in boxes_chains.keys()}

            torch.save(img_classes_dataset, gt_classes_filepath)
            torch.save(img_bboxes_dataset, gt_bboxes_filepath)

            return img_classes_dataset, img_bboxes_dataset

    def extract_boxes_and_chains(self):
        return generate_dataset(self.boxes_chains_filename, self.extract_boxes_and_chains_internal)

    def extract_boxes_and_chains_internal(self):
        extracted_chains = {}
        boxes_chains = {}
        box_count = 0
        self.log_print('Extracting bounding boxes and coreference chains...')
        for _, _, files in os.walk(self.bbox_dir_path):
            for filename in files:
                # Extract image file name from current file name
                image_id = int(filename.split('.')[0])

                # Extract bounding boxes from file
                bounding_boxes = []

                xml_filepath = os.path.join(self.bbox_dir_path, filename)
                xml_doc = minidom.parse(xml_filepath)
                for child_node in xml_doc.childNodes[0].childNodes:
                    # The bounding boxes are located inside a node named "object"
                    if child_node.nodeName == u'object':
                        # Go over all of the children of this node: if we find bndbox, this object is a bounding box
                        box_chain = None
                        for inner_child_node in child_node.childNodes:
                            if inner_child_node.nodeName == u'name':
                                box_chain = int(inner_child_node.childNodes[0].data)
                            if inner_child_node.nodeName == u'bndbox':
                                # This is a bounding box node
                                box_count += 1
                                bounding_box = [None, None, None, None]
                                for val_node in inner_child_node.childNodes:
                                    node_name = val_node.nodeName
                                    if node_name in self.coord_strs:
                                        coord_ind = self.coord_str_to_ind[node_name]
                                        bounding_box[coord_ind] = int(val_node.childNodes[0].data)

                                # Check that all coordinates were found
                                none_inds = [x for x in range(len(bounding_box)) if x is None]
                                bounding_box_ind = len(bounding_boxes)
                                if len(none_inds) > 0:
                                    for none_ind in none_inds:
                                        self.log_print('Didn\'t find coordinate ' + self.coord_strs[none_ind] +
                                                       ' for bounding box ' + str(bounding_box_ind) +
                                                       ' in image ' + filename)
                                    assert False
                                if box_chain is None:
                                    self.log_print('Didn\'t find chain for bounding box ' +
                                                   str(bounding_box_ind) + ' in image ' + filename)
                                    assert False
                                bounding_boxes.append((bounding_box, box_chain))

                                # Document chain
                                if box_chain not in extracted_chains:
                                    extracted_chains[box_chain] = True
                boxes_chains[image_id] = bounding_boxes

        self.log_print('Extracted bounding boxes and coreference chains')
        self.log_print('Found ' + str(len(boxes_chains)) + ' images, ' + str(box_count) + ' bounding boxes, ' + str(len(extracted_chains)) + ' coreference chains')
        chain_list = list(extracted_chains.keys())

        return boxes_chains, chain_list

    def get_chain_to_class_mapping(self, chain_list):
        return generate_dataset(self.chains_and_classes_filename, self.get_chain_to_class_mapping_internal, chain_list)

    def get_chain_to_class_mapping_internal(self, chain_list):
        chain_to_class_ind = {}
        class_str_to_ind = {}
        found_chains = {x: False for x in chain_list}
        print('Extracting chain to class mapping...')
        file_names = os.listdir(self.sentences_dir_path)
        for file_ind in range(len(file_names)):
            # Extract annotated sentences from file
            filename = file_names[file_ind]
            filepath = os.path.join(self.sentences_dir_path, filename)
            fp = open(filepath, 'r', encoding='utf-8')
            for line in fp:
                split_by_annotations = line.split('[/EN#')[1:]
                for line_part in split_by_annotations:
                    annotation = line_part.split()[0].split('/')
                    chain_ind = int(annotation[0])
                    class_str = annotation[1]

                    if class_str in class_str_to_ind:
                        class_ind = class_str_to_ind[class_str]
                    else:
                        class_ind = len(class_str_to_ind)
                        class_str_to_ind[class_str] = class_ind

                    chain_to_class_ind[chain_ind] = class_ind
                    if chain_ind in found_chains and not found_chains[chain_ind]:
                        found_chains[chain_ind] = True

        # Add the 'unknown' class for chains we couldn't find
        unknown_class_ind = len(class_str_to_ind)
        class_str_to_ind['unknown'] = unknown_class_ind
        chains_not_found = [x for x in chain_list if not found_chains[x]]
        for chain_ind in chains_not_found:
            chain_to_class_ind[chain_ind] = unknown_class_ind

        print('Extracted chain to class mapping')
        class_ind_to_str = {class_str_to_ind[k]: k for k in class_str_to_ind.keys()}

        return chain_to_class_ind, class_ind_to_str

    def get_class_mapping(self):
        _, chain_list = self.extract_boxes_and_chains()
        _, class_ind_to_str = self.get_chain_to_class_mapping(chain_list)
        return class_ind_to_str

    def get_image_path(self, image_id, slice_str):
        image_filename = str(image_id) + '.jpg'
        image_path = os.path.join(self.images_dir_path, image_filename)

        return image_path

