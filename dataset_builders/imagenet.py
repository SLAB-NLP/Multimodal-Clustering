##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

from utils.visual_utils import pil_image_trans
import torchvision.datasets as datasets
from dataset_builders.classification_dataset_builder import ClassificationDatasetBuilder


class MyImageNet(datasets.ImageNet):
    def __init__(self, root_dir_path, split, transform):
        super(MyImageNet, self).__init__(root=root_dir_path, split=split, transform=transform)

        one_word_classes = list(set([x[1] for x in self.class_to_idx.items() if ' ' not in x[0]]))
        self.samples = [x for x in self.samples if x[1] in one_word_classes]

    def __getitem__(self, idx):
        """ Imagenet original implementation returns list of two items: the first is the images, and the second is
        the labels. To fit our other datasets, we want a mapping instead of a list. """
        batch_list = datasets.ImageNet.__getitem__(self, idx)
        return {
            'image': batch_list[0],
            'label': batch_list[1]
        }


class ImageNet(ClassificationDatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(ImageNet, self).__init__(indent)

        self.root_dir_path = root_dir_path
        self.slices = ['train', 'val']

    def get_class_mapping(self):
        imagenet_class_to_idx_one_word = {x[0]: x[1] for x in self.imagenet_dataset.class_to_idx.items()
                                          if ' ' not in x[0]}
        class_mapping = {imagenet_class_to_idx_one_word[x]: x for x in imagenet_class_to_idx_one_word.keys()}

        return class_mapping

    def build_dataset(self, config=None):
        if config.slice_str not in self.slices:
            self.log_print('No such data slice: ' + str(config.slice_str) +
                           '. Please specify one of ' + str(self.slices))
            assert False

        imagenet_dataset = MyImageNet(self.root_dir_path, split='val', transform=pil_image_trans)
        self.imagenet_dataset = imagenet_dataset

        return imagenet_dataset
