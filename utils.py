import os
from PIL import Image
import numpy as np
from collections import defaultdict, Counter

class PairsDataRead:
    def __init__(self):
        with open('./Data/pairs.txt', 'r') as f:
            pairs_data = f.readlines()
        self.pairs_data = self.clean_pairs_data(pairs_data)
        with open('./Data/pairsDevTest.txt', 'r') as f:
            pairs_data_dev = f.readlines()
        self.pairs_data_dev = self.clean_pairs_data(pairs_data_dev)
        with open('./Data/pairsDevTrain.txt', 'r') as f:
            pairs_data_train = f.readlines()
        self.pairs_data_train = self.clean_pairs_data(pairs_data_train)

    def clean_pairs_data(self, pairs_data):
        pairs_list = [pair.strip().split('\t') for pair in pairs_data]
        pairs_list_clean = []
        for pair in pairs_list:
            pair_clean = []
            if len(pair) == 3:
                pair_clean.append(pair[0])
                pair_clean.append(int(pair[1]))
                pair_clean.append(int(pair[2]))
            if len(pair) == 4:
                pair_clean.append(pair[0])
                pair_clean.append(int(pair[1]))
                pair_clean.append(pair[2])
                pair_clean.append(int(pair[3]))
            if pair_clean:
                pairs_list_clean.append(pair_clean)
        return pairs_list_clean

class ImageDataRead:
    '''
    Has function which gives us the images array and the images path list.
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    
    def get_image_data(self):
        images_data = []
        images_path_data = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as image:
                            image.load()
                            image_data = np.asarray(image, dtype = "int16")
                            images_data.append(image_data)
                            images_path_data.append(file_path)
                    except:
                        print(f"Unknown error has occured while reading file {file_path}")
        images_data_array = np.stack(images_data, axis = 0)
        return images_data_array, images_path_data


class LFWDatasetCache:
    '''
    Works by storing all the image pairs in memory. Will probably not work for large datsets
    '''
    def __init__(self, image_data_dir):
        self.pdr = PairsDataRead()
        self.idr = ImageDataRead(image_data_dir)
        images_data_array, images_path_data = self.idr.get_image_data()
        self.image_1_data, self.image_2_data, self.labels = self._process_image_data(images_data_array, images_path_data)

    def _process_image_data(self, images_data_array, images_path_data):
        image_index_dict = defaultdict(lambda : None)
        character_index_dict = defaultdict(lambda : [])
        for idx, image_path in enumerate(images_path_data):
            image_name = image_path.split('\\')[-1]
            character_name = image_path.split('\\')[-2]
            image_index = idx
            image_index_dict[image_name] = image_index
            character_index_dict[character_name].append(image_index)
        
        image_1_index = [image_index_dict[get_image_file_n(pair[0], pair[1])] for pair in pdr.pairs_data_train]
        image_2_index = [
            image_index_dict[get_image_file_n(pair[0], pair[2])] if len(pair) == 3 
            else image_index_dict[get_image_file_n(pair[2], pair[3])] for pair in pdr.pairs_data_train
        ]
        matching_label = [
            1 if len(pair) == 3
            else 0 
            for pair in pdr.pairs_data_train
        ]
        image_1_data = images_data_array[image_1_index]
        image_2_data = images_data_array[image_2_index]
        labels = np.array(matching_label)
        return image_1_data, image_2_data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_1_data[idx], self.image_2_data[idx], self.labels[idx]


class LFWDataset:
    '''
    A more generic and slow implementation. Can work with extremely large datasets. Also has support for custom transformations.
    '''
    def __init__(self, image_data_dir, train = True, cache = True, transform = None, label_transform = None):
        self.pairs_data = PairsDataRead()
        self.image_data_dir = image_data_dir
        self.train = train
        self.cache = cache
        self.transform = transform
        self.label_transform = label_transform
    
    def __len__(self):
        if self.train:
            return len(self.pairs_data.pairs_data_train)
        else:
            return len(self.pairs_data.pairs_data_dev)

    def _get_image_from_name_idx(self, name, idx):
        idx_str = str(idx)
        idx_str = '0'*(4 - len(idx_str)) + idx_str
        file_name = f"{name}_{idx_str}.jpg"
        file_path = f"{self.image_data_dir}\{name}\{file_name}"
        img = Image.open(file_path)
        return np.array(img)

    
    def _get_image_label_pair(self, pair):
        if len(pair) == 3:
            label = 1
            img_1_name = pair[0]
            img_2_name = pair[0]
            img_1_idx = pair[1]
            img_2_idx = pair[2]
        else:
            label = 0
            img_1_name = pair[0]
            img_2_name = pair[2]
            img_1_idx = pair[1]
            img_2_idx = pair[3]
        img_1 = self._get_image_from_name_idx(img_1_name, img_1_idx)
        img_2 = self._get_image_from_name_idx(img_2_name, img_2_idx)
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.label_transform:
            label = self.label_transform(label)
        
        # img_1 = img_1.transpose(2, 0, 1)
        # img_2 = img_2.transpose(2, 0, 1)
        # img_1 = img_1.astype(np.float32)
        # img_2 = img_2.astype(np.float32)
        # label = np.array(label).astype(np.float32)
        return img_1, img_2, label

        
    def __getitem__(self, idx):
        if self.train:
            pairs_list = self.pairs_data.pairs_data_train
        else:
            pairs_list = self.pairs_data.pairs_data_dev
        pair = pairs_list[idx]
        return self._get_image_label_pair(pair)

        

