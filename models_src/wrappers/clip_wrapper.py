##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.


import clip
import torch
from loggable_object import LoggableObject


class CLIPWrapper(LoggableObject):
    """ This class wraps the CLIP model, described in the paper "Learning Transferable Visual Models From Natural
        Language Supervision" By Radford et al.
    """

    def __init__(self, class_mapping, indent):
        super(CLIPWrapper, self).__init__(indent)
        self.model, _ = clip.load('RN50', self.device)
        self.class_mapping = class_mapping

    """ Create the class embedding mat: for each class we create a relevant prompt, and encode it.
        The self.class_mapping dictionary maps the indices of ground-truth classes to their names. The class indices are
        not necessarily sequential, so we create a mat_ind_to_class_ind dictionary, which maps the number of row in the
        embedding matrix to the relevant ground-truth class index.
    """

    def initialize(self):
        prompts = {x: 'a photo of a ' + self.class_mapping[x] for x in self.class_mapping.keys()
                   if ' ' not in self.class_mapping[x]}
        class_ind_and_embedding = [(x[0], self.model.encode_text(clip.tokenize(x[1]).to(self.device)).float())
                                   for x in prompts.items()]
        self.class_embedding_mat = torch.cat([x[1] for x in class_ind_and_embedding])
        self.class_embedding_mat = self.class_embedding_mat / self.class_embedding_mat.norm(dim=-1, keepdim=True)
        self.mat_ind_to_class_ind = {i: class_ind_and_embedding[i][0] for i in range(len(class_ind_and_embedding))}

    """ Use the image encoder to encode input images. """

    def encode_image(self, visual_inputs):
        return self.model.encode_image(visual_inputs).float()

    """ Calculate the similarity of input image features.
        Image features are the output of the visual encoder on multiple images. For each image, we calculate the cosine
        similarity of its encoding with each class. So given N input images and M ground-truth classes, the output would
        be an NXM matrix, where the (i,j)th entry is the cosine similarity of the ith image and the jth ground-truth
        class.
    """

    def calculate_similarity(self, image_features):
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity_mat = norm_image_features @ self.class_embedding_mat.T

        return similarity_mat

    """ Get the number of ground-truth classes. """

    def get_class_num(self):
        return self.class_embedding_mat.shape[0]
