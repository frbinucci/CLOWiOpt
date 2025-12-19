#  The MIT License (MIT)
#  Copyright © 2025 University of Perugia
#
#  Author:
#
#  - Francesco Binucci (francesco.binucci@cnit.it)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Dataset class
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import ToTensor


class CarSegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_label=['car','cargroup'], transform=ToTensor(), target_size=(256, 256),data_format='jpg',**kwargs):
        """
        Args:
            images_dir (str): Directory containing images.
            labels_dir (str): Directory containing JSON labels.
            target_label (str): The object label to segment (default: 'car').
            transform (callable, optional): Optional transform to be applied on an image.
            target_size (tuple): Size to which the images will be resized (width, height).
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_label = target_label
        self.transform = transform
        self.target_size = target_size
        self.data_list = list()

        self.debug_mode = kwargs.get('debug_mode',True)
        self.break_point = kwargs.get('break_point',100)

        # Collect all image and json file paths
        self.image_filenames = []
        self.json_filenames = []
        for subdir, _, files in os.walk(self.images_dir):
            for filename in files:
                if filename.endswith(f'_leftImg8bit.{data_format}'):
                    self.image_filenames.append(os.path.join(subdir, filename))
                    # Corresponding JSON file name
                    json_filename = filename.replace(f'_leftImg8bit.{data_format}', '_gtCoarse_polygons.json')
                    json_path = os.path.join(self.labels_dir, os.path.relpath(subdir, self.images_dir), json_filename)
                    self.json_filenames.append(json_path)

        # Load image
        idx = 0
        for img_path in self.image_filenames:
            if "jpg" in img_path:

              image = Image.open(img_path).convert("RGB")  # Convert to RGB
              image = np.array(image)  # Convert to NumPy array
              # Resize image and mask
              original_width = image.shape[1]
              original_height = image.shape[0]
              image = cv2.resize(image, self.target_size)
              # Apply transformations (if any)
              if self.transform:
                  image = self.transform(image)
            else:
              image = np.load(img_path)
              image = torch.permute(torch.from_numpy(image),(2,1,0)).type(torch.float32)

            # Load corresponding JSON data
            json_path = self.json_filenames[idx]
            mask = self._create_mask(json_path, original_width, original_height)

            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)


            self.data_list.append([image,mask])

            # Dataset breaking (for debug purposes)
            if self.debug_mode == True:
                if idx==self.break_point:
                    break

            idx+=1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return  self.data_list[idx]


    def _create_mask(self, json_file, img_width, img_height):
        """
        Loads the JSON file and creates a binary mask for the target label.
        Args:
            json_file (str): Path to the JSON file.
            img_width (int): Original image width.
            img_height (int): Original image height.
        Returns:
            np.ndarray: Binary mask (0: background, 1: target objects).
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)  # Create a blank mask

        with open(json_file, 'r') as file:
            data = json.load(file)
            objects = data.get("objects", [])

            for obj in objects:
                if obj["label"].lower() in self.target_label:
                    # Extract polygon coordinates
                    polygon = np.array(obj["polygon"], dtype=np.int32)
                    cv2.fillPoly(mask, [polygon], 1)  # Fill the polygon in the mask
        return mask