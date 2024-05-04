import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

class LFWClassificationDataset:
    def __init__(self, data_dir="./Data", image_count_threshold=0, transforms=None, train=True):
        self.data_dir = data_dir
        self.image_count_threshold = image_count_threshold
        self.transforms = transforms
        self.train = train
        self.image_counts, self.image_path_data = self._get_image_counts()
        self._filter_data()
        self._split_data()
        self._encode_labels()

    def _get_image_counts(self):
        if not self.data_dir:
            return None, None
        image_path_data = defaultdict(lambda: None)
        image_counts = defaultdict(lambda: 0)
        for data_dir, sub_dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith("jpg"):
                    image_path = os.path.join(data_dir, file)
                    person_name = file[:-9]
                    image_counts[person_name] += 1
                    image_path_data[(person_name, image_counts[person_name])] = image_path
        return image_counts, image_path_data

    def _filter_data(self):
        self.filtered_image_paths = []
        self.filtered_person_labels = []
        for person_name, image_counts in self.image_counts.items():
            if image_counts >= self.image_count_threshold:
                for idx in range(1, image_counts + 1):
                    self.filtered_image_paths.append(self.image_path_data[(person_name, idx)])
                    self.filtered_person_labels.append(person_name)

    def _split_data(self):
        label_proc = [[label] for label in self.filtered_person_labels]
        self.train_image_paths, self.test_image_paths, self.train_person_labels, self.test_person_labels = train_test_split(
            self.filtered_image_paths,
            label_proc,
            shuffle=True,
            stratify=self.filtered_person_labels,
            test_size=0.3
        )

    def _encode_labels(self):
        self.oe = OrdinalEncoder()
        self.train_person_labels_encoded = self.oe.fit_transform(self.train_person_labels)
        self.test_person_labels_encoded = self.oe.transform(self.test_person_labels)

    def __getitem__(self, idx):
        if self.train:
            image_path = self.train_image_paths[idx]
            label = self.train_person_labels_encoded[idx]
        else:
            image_path = self.test_image_paths[idx]
            label = self.test_person_labels_encoded[idx]
        with Image.open(image_path) as img:
            image_array = np.array(img, dtype=np.float32)
        if self.transforms:
            image_array = self.transforms(image_array)
        return image_array, label

    def __len__(self):
        if self.train:
            return len(self.train_image_paths)
        return len(self.test_image_paths)
