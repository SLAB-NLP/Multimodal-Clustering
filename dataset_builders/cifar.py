##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import os
import abc
import torch
from utils.general_utils import generate_dataset
import pickle
from datasets_src.matrix_based_img_dataset import MatrixBasedImageDataset
from dataset_builders.classification_dataset_builder import ClassificationDatasetBuilder


class Cifar(ClassificationDatasetBuilder):

    def __init__(self, root_dir_path, data_dir_name, name, dict_key_mapping, indent):
        super(Cifar, self).__init__(indent)

        self.data_dir_path = os.path.join(root_dir_path, data_dir_name)
        self.slices = ['train', 'test']
        self.dict_key_mapping = dict_key_mapping

        self.cache_data_path_prefix = os.path.join(self.cached_dataset_files_dir, name + '_data_')

    @abc.abstractmethod
    def get_train_batch_file_paths(self):
        return

    @abc.abstractmethod
    def get_test_batch_file_paths(self):
        return

    @abc.abstractmethod
    def get_metadata_file_path(self):
        return

    def generate_data(self, slice_str):
        return generate_dataset(self.cache_data_path_prefix + slice_str, self.generate_data_internal, slice_str)

    def generate_data_internal(self, slice_str):
        if slice_str == 'train':
            batch_file_paths = self.get_train_batch_file_paths()
        elif slice_str == 'test':
            batch_file_paths = self.get_test_batch_file_paths()

        image_tensor_list = []
        label_list = []
        for batch_file_path in batch_file_paths:
            with open(batch_file_path, 'rb') as fo:
                data_key_name = self.dict_key_mapping['data']
                labels_key_name = self.dict_key_mapping['labels']

                batch_dict = pickle.load(fo, encoding='bytes')
                images_tensor = torch.from_numpy(batch_dict[data_key_name]).view(-1, 3, 32, 32)
                image_tensor_list.append(images_tensor)
                label_list += batch_dict[labels_key_name]
        data_tensor = torch.cat(image_tensor_list)

        return data_tensor, label_list

    def get_class_mapping(self):
        with open(self.get_metadata_file_path(), 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
            label_names_key_name = self.dict_key_mapping['label_names']

            class_mapping = {x: batch_dict[label_names_key_name][x].decode("utf-8") for x in
                             range(len(batch_dict[label_names_key_name]))}

        return class_mapping

    def build_dataset(self, config=None):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        image_tensor, label_list = self.generate_data(config.slice_str)
        class_mapping = self.get_class_mapping()

        return MatrixBasedImageDataset(image_tensor, label_list, class_mapping, config)


class Cifar10(Cifar):

    def __init__(self, root_dir_path, indent):
        dict_key_mapping = {
            'data': b'data',
            'labels': b'labels',
            'label_names': b'label_names'
        }
        super(Cifar10, self).__init__(root_dir_path, 'cifar-10-batches-py', 'cifar10', dict_key_mapping, indent)

        train_batch_file_name_prefix = 'data_batch_'
        train_batches = [1, 2, 3, 4, 5]
        self.train_batch_file_paths = [os.path.join(self.data_dir_path, train_batch_file_name_prefix + str(batch_ind))
                                       for batch_ind in train_batches]

        test_batch_file_name = 'test_batch'
        self.test_batch_file_paths = [os.path.join(self.data_dir_path, test_batch_file_name)]

        metadata_file_name = 'batches.meta'
        self.metadata_file_path = os.path.join(self.data_dir_path, metadata_file_name)

    def get_train_batch_file_paths(self):
        return self.train_batch_file_paths

    def get_test_batch_file_paths(self):
        return self.test_batch_file_paths

    def get_metadata_file_path(self):
        return self.metadata_file_path


class Cifar100(Cifar):

    def __init__(self, root_dir_path, indent):
        dict_key_mapping = {
            'data': b'data',
            'labels': b'fine_labels',
            'label_names': b'fine_label_names'
        }
        super(Cifar100, self).__init__(root_dir_path, 'cifar-100-python', 'cifar100', dict_key_mapping, indent)

        self.train_batch_file_paths = [os.path.join(self.data_dir_path, 'train')]
        self.test_batch_file_paths = [os.path.join(self.data_dir_path, 'test')]
        self.metadata_file_path = os.path.join(self.data_dir_path, 'meta')

    def get_train_batch_file_paths(self):
        return self.train_batch_file_paths

    def get_test_batch_file_paths(self):
        return self.test_batch_file_paths

    def get_metadata_file_path(self):
        return self.metadata_file_path
