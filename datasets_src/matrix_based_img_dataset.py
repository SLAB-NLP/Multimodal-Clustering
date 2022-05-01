##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import torch.utils.data as data
import torchvision.transforms as transforms
from utils.visual_utils import tensor_trans


class MatrixBasedImageDataset(data.Dataset):
    """A dataset with only image samples, that are stored in a single matrix, rather than in jpg images.
    """

    def __init__(self,
                 image_tensor,
                 label_list,
                 class_mapping,
                 config):
        self.image_tensor = image_tensor
        self.label_list = label_list
        self.class_mapping = class_mapping
        self.config = config

        mean_tuple = (0.48145466, 0.4578275, 0.40821073)
        std_tuple = (0.26862954, 0.26130258, 0.27577711)
        self.normalizer = transforms.Normalize(mean_tuple, std_tuple)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image_tensor = self.image_tensor[idx].float()
        label = self.label_list[idx]

        image_tensor = tensor_trans(image_tensor)

        sample = {
            'image': image_tensor,
            'label': label,
            'index': idx
        }

        return sample
