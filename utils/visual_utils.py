##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchcam.utils import overlay_mask
from models_src.underlying_models.simclr import SimCLRModel
import numpy as np


""" This file contains functions and definitions related to vision used across all the project. """

# All images are resized to a uniform size
wanted_image_size = (224, 224)
# Pixel values are normalized
mean_tuple = (0.48145466, 0.4578275, 0.40821073)
std_tuple = (0.26862954, 0.26130258, 0.27577711)
# A transformation from PIL Image to a tensor
pil_image_trans = transforms.Compose([
    transforms.Resize(wanted_image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_tuple, std_tuple)
])
# A transformation for a tensor
tensor_trans = transforms.Compose([
    transforms.Resize(wanted_image_size)
])
# Unnormalize the normalized pixels back, if we want to display the image later on
reverse_mean = ((-1)*mean_tuple[0]/std_tuple[0], (-1)*mean_tuple[1]/std_tuple[1], (-1)*mean_tuple[2]/std_tuple[2])
reverse_std = (1/std_tuple[0], 1/std_tuple[1], 1/std_tuple[2])
unnormalize_trans = transforms.Normalize(reverse_mean, reverse_std)

""" Functions related to extracting bounding boxes out of heatmaps. """


def find_largest_connected_component(input_tensor):
    """ Find the edges of the largest connected component (cc) in a 2D tensor, where each pixel is
    either valued 0 or 1. A (cc) is defined as a set of 1-valued pixels, where each pixel is a
    neighbor of some other pixel in the set, and none of the pixels has a 1-valued neighbor outside
    the set.
    We go over all the pixels, row-by-row, and inside each row column-by-column. For each pixel
    with value 1, we check if the previously visited neighbors (up-left, up, up-right, left) are
    already assigned with a cc. If not- we assign a new cc. If yes- we check if multiple neighbors
    have cc, and if they are assigned to different cc's- they are actually the same one (through the
    current pixel that connects both of them). So we map one of the cc's to the other, and will
    deal with this later.
    After we finish this traverse, we map each cc to its "final" cc: we follow the cc mapping until
    we reach a cc which is not mapped to any other cc. Then we calculate the size of each cc, to
    find the largest."""
    # First traverse all the pixels
    pixel_to_cc_ind, cc_to_final_cc, cc_num = map_pixel_to_cc(input_tensor)

    # Next, we need to create the final mapping
    cc_to_final_cc = create_final_cc_mapping(cc_num, cc_to_final_cc)
    # Change the pixel -> cc mapping to fit the final cc's we found
    pixel_to_cc_ind = \
        {x: cc_to_final_cc[pixel_to_cc_ind[x]] for x in pixel_to_cc_ind.keys()}

    # Next, find the largest cc
    largest_cc_ind = argmax_cc(pixel_to_cc_ind)

    # Now, find the largest cc's edges
    height = input_tensor.shape[0]
    width = input_tensor.shape[1]
    largest_cc_edges = find_cc_edges(pixel_to_cc_ind, height, width, largest_cc_ind)

    return largest_cc_edges


def map_pixel_to_cc(input_tensor):
    """ We go over all the pixels, row-by-row, and inside each row column-by-column. For each pixel
    with value 1, we check if the previously visited neighbors (up-left, up, up-right, left) are
    already assigned with a cc. If not- we assign a new cc. If yes- we check if multiple neighbors
    have cc, and if they are assigned to different cc's- they are actually the same one (through the
    current pixel that connects both of them). So we map one of the cc's to the other, and will
    deal with this later."""
    next_cc_ind = 0
    pixel_to_cc_ind = {}
    cc_to_final_cc = {}
    height = input_tensor.shape[0]
    width = input_tensor.shape[1]

    cur_row = 0
    cur_col = 0
    pixel_ind = 0  # The index of the pixel in a flattened image (ranges from 0 to width*height-1)
    while cur_row is not None:
        if input_tensor[cur_row, cur_col]:
            cc_ind = find_neighbors_cc_ind(pixel_to_cc_ind, cc_to_final_cc, width, pixel_ind)
            if cc_ind is None:
                # New connected component
                cc_ind = next_cc_ind
                next_cc_ind += 1
            pixel_to_cc_ind[pixel_ind] = cc_ind

        cur_row, cur_col = advance_pixel(cur_row, cur_col, height, width)
        pixel_ind += 1

    return pixel_to_cc_ind, cc_to_final_cc, next_cc_ind


def find_neighbors_cc_ind(pixel_to_cc_ind, cc_to_final_cc, width, pixel_ind):
    """ Find the cc indices of neighbour pixels. """

    # Need the check the neighbors we already visited: upper-left, upper, upper-right, left
    cc_inds = []
    if pixel_ind >= width:
        # This is not the first row
        if pixel_ind % width > 0:
            # This is not the first column, check upper-left neighbor
            upper_left_ind = pixel_ind - width - 1
            if upper_left_ind in pixel_to_cc_ind:
                cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_left_ind], cc_to_final_cc))
        # Check upper neighbor
        upper_ind = pixel_ind - width
        if upper_ind in pixel_to_cc_ind:
            cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_ind], cc_to_final_cc))

        if pixel_ind % width < width-1:
            # This is not the last column, check upper-right neighbor
            upper_right_ind = pixel_ind - width + 1
            if upper_right_ind in pixel_to_cc_ind:
                cc_inds.append(get_final_cc(pixel_to_cc_ind[upper_right_ind], cc_to_final_cc))

    if pixel_ind % width > 0:
        # This is not the first column, check left neighbor
        left_ind = pixel_ind - 1
        if left_ind in pixel_to_cc_ind:
            cc_inds.append(get_final_cc(pixel_to_cc_ind[left_ind], cc_to_final_cc))

    # If multiple cc's are found, update the cc mapping
    cc_inds = list(set(cc_inds))
    if len(cc_inds) > 0:
        cc_ind = cc_inds[0]  # Arbitrarily choose the first one to be the one all the others are mapped to
        for other_cc_ind in cc_inds[1:]:
            cc_to_final_cc[other_cc_ind] = cc_ind
    else:
        cc_ind = None

    return cc_ind


def get_final_cc(cc_ind, cc_to_final_cc):
    """ When we find during the first traverse that two cc's are actually one, we map one of them to the other.
        Finally, to find the cc to which a cc will be mapped, we need to following the mapping until we get to a cc not
        mapped to any other cc.
    """
    while cc_ind in cc_to_final_cc:
        cc_ind = cc_to_final_cc[cc_ind]
    return cc_ind


def advance_pixel(row, col, h, w):
    """ Given a pixel location and the dimensions of the image (height and width), give the next pixel in a traverse
        going from left to right and then from top to bottom.
    """
    if col < w - 1:
        return row, col + 1
    else:
        if row < h - 1:
            return row + 1, 0
        else:
            return None, None


def create_final_cc_mapping(cc_num, cc_to_final_cc):
    """ After mapping cc's to each other, we produce the final cc's by following each cc mapping and mapping it to the
        cc not mapped to any other cc.
    """
    return {x: get_final_cc(x, cc_to_final_cc) for x in range(cc_num)}


def argmax_cc(pixel_to_cc_ind):
    """ After we assigned each pixel to cc, find the largest cc. """
    cc_ind_to_size = {}
    for _, cc_ind in pixel_to_cc_ind.items():
        if cc_ind not in cc_ind_to_size:
            cc_ind_to_size[cc_ind] = 0
        cc_ind_to_size[cc_ind] += 1
    largest_cc_size = 0
    largest_cc_ind = 0
    for cc_ind, cc_size in cc_ind_to_size.items():
        if cc_size > largest_cc_size:
            largest_cc_size = cc_size
            largest_cc_ind = cc_ind

    return largest_cc_ind


def find_cc_edges(pixel_to_cc_ind, height, width, wanted_cc_ind):
    """ Find the edges of a given cc.
        The top edge is the location of the highest pixel, and the left, right and bottom edges are defined equivalently.
    """
    cc_pixels = []
    for pixel_ind, cc_ind in pixel_to_cc_ind.items():
        if cc_ind == wanted_cc_ind:
            cc_pixels.append((pixel_ind // width, pixel_ind % width))
    upper_edge = height
    lower_edge = 0
    left_edge = width
    right_edge = 0
    for row, col in cc_pixels:
        if row < upper_edge:
            upper_edge = row
        if row > lower_edge:
            lower_edge = row
        if col < left_edge:
            left_edge = col
        if col > right_edge:
            right_edge = col

    return upper_edge, lower_edge, left_edge, right_edge


def calc_ious(box1, box2):
    """ Calculate intersection-over-union (IoU) between pairs of bounding boxes. """
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area_sum = box1_area.reshape((-1, 1)) + box2_area.reshape((1, -1))

    intersection_x_min = torch.max(box1[:, 0].reshape((-1, 1)), box2[:, 0].reshape((1, -1)))
    intersection_x_max = torch.min(box1[:, 2].reshape((-1, 1)), box2[:, 2].reshape((1, -1)))
    intersection_y_min = torch.max(box1[:, 1].reshape((-1, 1)), box2[:, 1].reshape((1, -1)))
    intersection_y_max = torch.min(box1[:, 3].reshape((-1, 1)), box2[:, 3].reshape((1, -1)))

    intersection_width = torch.max(intersection_x_max - intersection_x_min, torch.tensor([0.]))
    intersection_height = torch.max(intersection_y_max - intersection_y_min, torch.tensor([0.]))
    intersection_area = intersection_width * intersection_height
    union_area = area_sum - intersection_area
    ious = intersection_area / union_area
    return ious


def resize_activation_map(activation_map):
    """ Resize the given activation map from its current size to the original image size.
        Current size of the activation map is the size of the last layer of the visual encoder.
    """
    upsample_op = nn.Upsample(size=wanted_image_size, mode='bicubic', align_corners=False)
    activation_map = activation_map.view(1, 1, activation_map.shape[0], activation_map.shape[1])
    return upsample_op(activation_map).view(wanted_image_size)


def predict_bbox(activation_map, segment_threshold_rate=0.5):
    """ Predict a bounding box given an activation map, according to the method described in the paper
        "Learning Deep Features for Discriminative Localization" by Zhou et al.
        We find the maximal value in the activation map, and set the segment threshold to this value multiplied by
        the given segment_threshold_rate. We mark every pixel with a value that exceeds the threshold with 1, and all
        others with 0. Then, we find the largest connected component of pixels marked 1.
    """
    orig_image_activation_vals = resize_activation_map(activation_map)

    cam_max_val = torch.max(orig_image_activation_vals)

    segmentation_map = torch.zeros(wanted_image_size, dtype=torch.bool)
    segmentation_map[orig_image_activation_vals >= segment_threshold_rate * cam_max_val] = True

    upper_edge, lower_edge, left_edge, right_edge = \
        find_largest_connected_component(segmentation_map)

    bbox = [left_edge, upper_edge, right_edge, lower_edge]

    return bbox


def predict_bboxes_with_activation_maps(activation_maps):
    """ Predict a list of bounding boxes given a list of activation maps. """
    predicted_bboxes = []
    for sample_activation_maps in activation_maps:
        predicted_bboxes.append([])
        for predicted_class_activation_map in sample_activation_maps:
            bbox = predict_bbox(predicted_class_activation_map)
            predicted_bboxes[-1].append(bbox)

    return predicted_bboxes


def get_resized_gt_bboxes(gt_bboxes, orig_image_size):
    """ The provided ground-truth boxes are relevant for the original image size.
        However, we resized the image to the wanted_image_size. This function returns the resized bounding boxes,
        matched to the wanted_image_size.
    """
    rel_gt_bboxes = [[
        x[0] / orig_image_size[0],
        x[1] / orig_image_size[1],
        x[2] / orig_image_size[0],
        x[3] / orig_image_size[1]
    ] for x in gt_bboxes]

    resized_gt_bboxes = [[
        int(x[0] * wanted_image_size[0]),
        int(x[1] * wanted_image_size[1]),
        int(x[2] * wanted_image_size[0]),
        int(x[3] * wanted_image_size[1])
    ] for x in rel_gt_bboxes]

    return resized_gt_bboxes


def plot_heatmap(image_tensor, activation_map, explicitly_upsample):
    """ Plot a heatmap on the given image_tensor given the provided activation_map.
        if explicitly_upsample == True, we use the nn.Upsample function to upsample, otherwise we use PIL implicit
        upsampling.
    """
    trans = transforms.ToPILImage()
    image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3])
    image_obj = trans(image_tensor)
    image_obj = image_obj.resize(wanted_image_size)
    if explicitly_upsample:
        heatmap = to_pil_image(resize_activation_map(activation_map), mode='F')
    else:
        heatmap = to_pil_image(activation_map, mode='F')
    result = overlay_mask(image_obj, heatmap)

    return result


""" Other functions. """


def get_image_shape_from_id(image_id, get_image_path_func, slice_str):
    """ Get the original image size given it's id. """
    image_obj = Image.open(get_image_path_func(image_id, slice_str))
    return np.array(image_obj).shape


def generate_visual_model(model_str, output_size, pretrained_base):
    """ Generate a visual encoder.
        model_str: The name of the wanted model.
        output_size: The size of the output layer.
        pretrained_base: Whether we should initialize with ImageNet pretrained weights.
    """
    if model_str == 'resnet18':
        model = models.resnet18(pretrained=pretrained_base)
        model.fc = nn.Linear(512, output_size)
    elif model_str == 'resnet34':
        model = models.resnet34(pretrained=pretrained_base)
        model.fc = nn.Linear(512, output_size)
    elif model_str == 'resnet50':
        model = models.resnet50(pretrained=pretrained_base)
        model.fc = nn.Linear(2048, output_size)
    elif model_str == 'resnet101':
        model = models.resnet101(pretrained=pretrained_base)
        model.fc = nn.Linear(2048, output_size)
    elif model_str == 'vgg16':
        model = models.vgg16(pretrained=pretrained_base)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(512, output_size)
    elif model_str == 'googlenet':
        model = models.googlenet(pretrained=pretrained_base, aux_logits=False)
        model.fc = nn.Linear(1024, output_size)
    elif model_str == 'simclr':
        model = SimCLRModel(output_encoder=False)
        model.g = nn.Linear(2048, output_size)

    return model
