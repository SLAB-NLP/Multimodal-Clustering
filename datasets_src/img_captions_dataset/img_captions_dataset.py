##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch.utils.data as data
from utils.text_utils import prepare_data
import abc


class ImageCaptionDataset(data.Dataset):
    """ The base class for datasets containing images and corresponding captions. """

    def __init__(self, config):
        super(ImageCaptionDataset, self).__init__()
        self.config = config

    """ Tokenize a list of captions. """

    def prepare_data(self, captions):
        return prepare_data(captions, lemmatize=self.config.lemmatize)

    """ Get a mapping from token to its count in the dataset. """

    def get_token_count(self):
        token_count = {}
        caption_data = self.get_caption_data()
        i = 0
        for x in caption_data:
            token_list = self.prepare_data([x['caption']])[0]
            for token in token_list:
                if token not in token_count:
                    token_count[token] = 0
                token_count[token] += 1
            i += 1

        return token_count
    
    """ Get the caption data: A list of dictionaries that contain image id and a corresponding caption. For example:
        [
            {'image_id': 123, 'caption': 'A large dog'},
            {'image_id': 456, 'caption': 'A white airplane'},
            ...
        ]
    """

    @abc.abstractmethod
    def get_caption_data(self):
        return
