##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################

import os
import torch
import json
from dataset_builders.img_caption_dataset_builder import ImageCaptionDatasetBuilder
from utils.general_utils import generate_dataset


class WikiScenes(ImageCaptionDatasetBuilder):

    def __init__(self, root_dir_path, indent):
        super(WikiScenes, self).__init__('WikiScenes', indent)
        self.root_dir_path = root_dir_path

        self.cathedral_dir = os.path.join(self.root_dir_path, 'cathedrals')
        self.category_filename = 'category.json'

        self.image_id_to_path_file_path = os.path.join(self.cached_dataset_files_dir, self.name + '_image_id_to_name')
        if os.path.isfile(self.image_id_to_path_file_path):
            self.image_id_to_path = torch.load(self.image_id_to_path_file_path)
        else:
            self.image_id_to_path = {}

        self.wanted_suffixes = ['jpg', 'jpeg', 'png']

        self.damaged_image_ids = [1002002005005154]

    def generate_caption_data_internal(self, slice_str):
        return self.generate_captions_and_gt_classes_data()[0]

    def generate_data_for_class(self, path, category_ind, id_prefix):
        # Check if this level contains pictures
        if 'pictures' in os.listdir(path):
            # self.adjust_image_file_names(path, id_prefix)
            category_file_path = os.path.join(path, 'category.json')
            with open(category_file_path, encoding='utf8') as category_fp:
                category_data = json.load(category_fp)
                image_info_list = list(category_data.items())

                # Filter images that doesn't really exist
                picture_names = os.listdir(os.path.join(path, 'pictures'))
                image_info_list = [x for x in image_info_list if x[0] in picture_names]

                # Filter images with unwanted suffix
                image_info_list = [x for x in image_info_list if x[0].split('.')[-1].lower() in self.wanted_suffixes]

                # DELETE- filter images with very long names
                image_info_list = [x for x in image_info_list if os.path.isfile(os.path.join(path, 'pictures', x[0]))]

                image_num = len(image_info_list)
                assert image_num <= self.image_id_segment_size
                image_ids = [id_prefix * self.image_id_segment_size + x for x in range(image_num)]

                # Filter damaged images
                damaged_image_ids_indices = [i for i in range(image_num) if image_ids[i] in self.damaged_image_ids]
                image_info_list = [image_info_list[i] for i in range(image_num) if i not in damaged_image_ids_indices]
                image_ids = [image_ids[i] for i in range(image_num) if i not in damaged_image_ids_indices]
                image_num = len(image_info_list)

                self.image_id_to_name.update({image_ids[i]: image_info_list[i][0] for i in range(image_num)})
                self.caption_data += [{'image_id': image_ids[i], 'caption': image_info_list[i][1]['caption']}
                                      for i in range(image_num)]
                self.gt_classes_data.update({image_ids[i]: [category_ind] for i in range(image_num)})

        # Run recursively
        assert len(os.listdir(path)) <= self.image_id_segment_size
        for loc in os.listdir(path):
            new_path = os.path.join(path, loc)
            if os.path.isdir(new_path) and loc.isnumeric():
                new_id_prefix = self.image_id_segment_size * id_prefix + int(loc)
                self.generate_data_for_class(new_path, category_ind, new_id_prefix)

    def adjust_image_file_names(self, path, id_prefix):
        """ This function is meant to solve an issue created by file name length.
        Some of the files in the dataset have very long file names, with which windows
        can't deal. So we replace the file names with short file names, i.e., the relevant
        image id. """
        new_category_data = {}
        category_file_path = os.path.join(path, 'category.json')
        with open(category_file_path, encoding='utf8') as category_fp:
            old_category_data = json.load(category_fp)
            image_info_list = list(old_category_data.items())
            image_num = len(image_info_list)
            image_ids = [id_prefix * self.image_id_segment_size + x for x in range(image_num)]
            picutres_dir_path = os.path.join(path, 'pictures')
            old_cwd = os.getcwd()
            os.chdir(picutres_dir_path)

            for file_ind in range(image_num):
                old_file_name = image_info_list[file_ind][0]
                if not old_file_name.endswith('.jpg'):
                    print('***File name doesn\'t end with jpg! file name: ' + str(old_file_name))
                new_file_name = str(image_ids[file_ind]) + '.jpg'

                os.rename(old_file_name, new_file_name)

                new_category_data[new_file_name] = old_category_data[old_file_name]

            os.chdir(old_cwd)

        with open(category_file_path, 'w', encoding='utf8') as category_fp:
            json.dump(new_category_data, category_fp)

    def generate_gt_classes_data(self, slice_str):
        return generate_dataset(self.file_paths[slice_str]['gt_classes'],
                                self.generate_gt_classes_data_internal, slice_str)

    def generate_gt_classes_data_internal(self, slice_str):
        self.generate_captions_and_gt_classes_data()[1]

    def generate_gt_bboxes_data(self, slice_str):
        return {}

    def generate_gt_classes_bboxes_data(self, slice_str):
        return self.generate_gt_classes_data(slice_str), self.generate_gt_bboxes_data(slice_str)

    def generate_captions_and_gt_classes_data(self):
        caption_filepath = self.file_paths['train']['captions']
        gt_classes_filepath = self.file_paths['train']['gt_classes']

        if os.path.exists(caption_filepath):
            return torch.load(caption_filepath), torch.load(gt_classes_filepath)
        else:
            self.caption_data = []
            self.gt_classes_data = {}

            for loc in os.listdir(self.cathedral_dir):
                path = os.path.join(self.cathedral_dir, loc)
                if os.path.isdir(path):
                    category_ind = int(loc)
                    initial_id_prefix = self.image_id_segment_size + category_ind
                    self.generate_data_for_class(path, category_ind, initial_id_prefix)

            torch.save(self.caption_data, caption_filepath)
            torch.save(self.gt_classes_data, gt_classes_filepath)
            torch.save(self.image_id_to_name, self.image_id_to_name_file_path)

            return self.caption_data, self.gt_classes_data

    def get_class_mapping(self):
        main_category_filename = os.path.join(self.cathedral_dir, self.category_filename)
        with open(main_category_filename, encoding='utf8') as main_category_fp:
            category_data = json.load(main_category_fp)
            class_name_to_ind = category_data['pairs']
            class_mapping = {int(x[1]): x[0] for x in class_name_to_ind.items()}

        return class_mapping

    def get_image_path(self, image_id, slice_str):
        image_path = os.path.join('pictures', self.image_id_to_name[image_id])
        image_id //= self.image_id_segment_size
        while image_id != 1:
            cur_dir_name = str(image_id % self.image_id_segment_size)
            image_path = os.path.join(cur_dir_name, image_path)
            image_id //= self.image_id_segment_size

        return os.path.join(self.cathedral_dir, image_path)
