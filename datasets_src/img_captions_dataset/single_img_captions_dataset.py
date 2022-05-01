##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
from utils.visual_utils import pil_image_trans
from datasets_src.img_captions_dataset.img_captions_dataset import ImageCaptionDataset
from PIL import Image


class SingleImageCaptionDataset(ImageCaptionDataset):
    """ A dataset with image samples and corresponding captions.
        The dataset is comprised of pairs of captions and images.
        The captions are stored in the dataset from the beginning, the images are loaded online during the __getitem__
        function execution. Until they are loaded we only keep the image id, and when we need to load the image- we
        use the provided get_image_path_func which is expected to provide us the path to the image, given its id.
        The dataset also provides, optionally, the ground-truth classes of each image, and the ground-truth bounding
        boxes of each image (these options are specified in the configuration).
        The class mapping is a mapping from class indices to class names.
        Finally, if the user provides the data_indices list, we only take the samples with indices in this list.
    """

    def __init__(self,
                 caption_file,
                 gt_classes_file,
                 class_mapping,
                 get_image_path_func,
                 config,
                 val_data_indices=None):
        super(SingleImageCaptionDataset, self).__init__(config)

        self.caption_data = torch.load(caption_file)

        if self.config.exclude_val_data:
            val_data_dic = {i: True for i in val_data_indices}
            sample_num = len(self.caption_data)
            self.caption_data = [self.caption_data[i] for i in range(sample_num) if i not in val_data_dic]
        if self.config.only_val_data:
            self.caption_data = [self.caption_data[i] for i in val_data_indices]

        if gt_classes_file is None:
            self.gt_classes_data = None
        else:
            self.gt_classes_data = torch.load(gt_classes_file)
        # We need the ground-truth class data to simplify captions
        assert (not config.simplified_captions) or (self.gt_classes_data is not None)

        self.class_mapping = class_mapping
        self.get_image_path_func = get_image_path_func

    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_caption_data = self.caption_data[idx]

        # Load image
        image_id = item_caption_data['image_id']
        image_obj = Image.open(self.get_image_path_func(image_id, self.config.slice_str))
        orig_image_size = image_obj.size
        image_tensor = pil_image_trans(image_obj)

        # Caption
        if self.config.simplified_captions:
            caption = ' '.join([self.class_mapping[x] for x in self.gt_classes_data[image_id]])
        else:
            caption = item_caption_data['caption']

        sample = {
            'image_id': image_id,
            'image': image_tensor,
            'orig_image_size': orig_image_size,
            'caption': caption,
        }

        return sample

    def get_caption_data(self):
        return self.caption_data
