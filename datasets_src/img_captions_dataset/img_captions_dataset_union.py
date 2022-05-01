##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import torch
from datasets_src.img_captions_dataset.img_captions_dataset import ImageCaptionDataset


class ImageCaptionDatasetUnion(ImageCaptionDataset):
    """An implementation of union of several datasets of the ImageCaptionDataset class.
    Currently not implemented with gt classes or bboxes.
    """

    def __init__(self, dataset_list, config):
        super(ImageCaptionDatasetUnion, self).__init__(config)
        self.dataset_list = dataset_list
        self.dataset_sizes = [len(dataset) for dataset in self.dataset_list]
        self.sample_indices = []
        for i in range(len(self.dataset_sizes)):
            self.sample_indices += [(i, x) for x in range(self.dataset_sizes[i])]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataset_index, sample_index = self.sample_indices[idx]
        return self.dataset_list[dataset_index].__getitem__(sample_index)

    def get_caption_data(self):
        caption_data = []
        for dataset in self.dataset_list:
            caption_data += dataset.get_caption_data()

        return caption_data
