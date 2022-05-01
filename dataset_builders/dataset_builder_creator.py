##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.cifar import Cifar10, Cifar100
from dataset_builders.imagenet import ImageNet
from dataset_builders.coco import Coco
from dataset_builders.flickr30 import Flickr30
from dataset_builders.pascal_voc import PascalVOC
from dataset_builders.wiki_scenes import WikiScenes


""" This utility creates a dataset builder given the dataset name.
    A dataset builder is an object that builds a torch.utils.data.Data object (the actual dataset), given the external
    files of the dataset.
    The utility assumes the name of the dataset is the name of its root directory.
"""


def create_dataset_builder(dataset_name):
    root_dir = os.path.join(DatasetBuilder.get_datasets_dir(), dataset_name)
    if dataset_name == 'cifar-10':
        dataset_generator = Cifar10(root_dir, 1)
        val_slice_str = 'test'
        multi_label = False
    elif dataset_name == 'cifar100':
        dataset_generator = Cifar100(root_dir, 1)
        val_slice_str = 'test'
        multi_label = False
    elif dataset_name == 'ImageNet':
        dataset_generator = ImageNet(root_dir, 1)
        val_slice_str = 'val'
        multi_label = False
    elif dataset_name == 'COCO':
        dataset_generator = Coco(root_dir, 1)
        val_slice_str = 'test'
        multi_label = True
    elif dataset_name == 'flickr30':
        dataset_generator = Flickr30(root_dir, 1)
        val_slice_str = 'val'
        multi_label = True
    elif dataset_name == 'VOC2012':
        dataset_generator = PascalVOC(root_dir, 1)
        val_slice_str = ''
        multi_label = True
    elif dataset_name == 'WikiScenes':
        dataset_generator = WikiScenes(root_dir, 1)
        val_slice_str = ''
        multi_label = True
    else:
        assert False

    return dataset_generator, val_slice_str, multi_label
