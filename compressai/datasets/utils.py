# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import rawpy
import glob
import numpy as np
import copy


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root_raw, root_srgb, transform=None, key="train"):
        if key == "train":
            self.key = 'train'
            self.samples_raw = sorted(glob.glob(root_raw + '/train/*.dng'))
            self.samples_srgb = sorted(glob.glob(root_srgb + '/train/*.png'))
        else:
            self.samples_raw = sorted(glob.glob(root_raw + '/test/*.dng'))
            self.samples_srgb = sorted(glob.glob(root_srgb + '/test/*.png'))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        # raw image 읽어와서 float로 변환 --> 0~255로 normalize
        # img_raw = rawpy.imread(self.samples_raw[index]).raw_image.astype(np.float32) / (2**14-1) * 255
        # img_raw = Image.fromarray(np.clip(np.round(img_raw), 0, 255).astype('uint8'))
        # img_srgb = Image.open(self.samples_srgb[index]).convert("RGB")

        # raw image 읽어와서 float 변환 --> 0~1로 normalize
        img_raw = rawpy.imread(self.samples_raw[index]).raw_image.astype(np.float32) / (2**14-1)
        img_srgb = np.array(Image.open(self.samples_srgb[index]).convert("RGB")).astype(np.float32) / (2**8-1)

        if self.transform:
            return {'raw': self.transform(copy.deepcopy(img_raw)),
                    'srgb': self.transform(copy.deepcopy(img_srgb))}
        return {'raw': img_raw,
                'srgb': img_srgb}

    def __len__(self):
        return len(self.samples_raw)

