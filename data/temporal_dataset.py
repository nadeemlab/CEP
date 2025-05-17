import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np


class TemporalDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A0 = os.path.join(opt.dataroot, opt.phase + 'A0')  # create a path '/path/to/data/trainA'
        self.dir_B0 = os.path.join(opt.dataroot, opt.phase + 'B0')  # create a path '/path/to/data/trainB'
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A1')  # create a path '/path/to/data/trainA'
        self.dir_B1 = os.path.join(opt.dataroot, opt.phase + 'B1')  # create a path '/path/to/data/trainB'
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')  # create a path '/path/to/data/trainA'
        self.dir_B2 = os.path.join(opt.dataroot, opt.phase + 'B2')  # create a path '/path/to/data/trainB'

        self.A0_paths = sorted(make_dataset(self.dir_A0, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B0_paths = sorted(make_dataset(self.dir_B0, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A1_paths = sorted(make_dataset(self.dir_A1, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B1_paths = sorted(make_dataset(self.dir_B1, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B2_paths = sorted(make_dataset(self.dir_B2, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        assert len(self.A0_paths) == len(self.A1_paths) and len(self.A0_paths) == len(self.A2_paths), "Unequal trainA dataset length"
        self.A_size = len(self.A0_paths)  # get the size of dataset A

        assert len(self.B0_paths) == len(self.B1_paths) and len(self.B0_paths) == len(self.B2_paths), "Unequal trainB dataset length"
        self.B_size = len(self.B0_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.input_nc = input_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        #get A0 and A1 image paths
        A0_path = self.A0_paths[index % self.A_size]  
        A1_path = self.A1_paths[index % self.A_size]  
        A2_path = self.A2_paths[index % self.A_size]  

        #get index for B
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        B0_path = self.B0_paths[index_B]
        B1_path = self.B1_paths[index_B]
        B2_path = self.B2_paths[index_B]


        A0_img = Image.open(A0_path).convert('RGB')
        A1_img = Image.open(A1_path).convert('RGB')
        A2_img = Image.open(A2_path).convert('RGB')


        B0_img = Image.open(B0_path).convert('RGB')
        B1_img = Image.open(B1_path).convert('RGB')
        B2_img = Image.open(B2_path).convert('RGB')


        #get transformation for image A
        transform_params = get_params(self.opt, A0_img.size)
        transform_A = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        # apply image transformation
        A0 = transform_A(A0_img)
        A1 = transform_A(A1_img)
        A2 = transform_A(A2_img)


        #get transformation for image B
        transform_params = get_params(self.opt, B0_img.size)
        transform_B = get_transform(self.opt, transform_params, grayscale=False)

        #apply image transformation
        B0 = transform_B(B0_img)
        B1 = transform_B(B1_img)
        B2 = transform_B(B2_img)


        return {'A0': A0, 'A1': A1, 'A2': A2, 'B0': B0, 'B1': B1, 'B2': B2,  'A0_paths': A0_path,'A1_paths': A1_path, 
                'A2_paths': A2_path, 'B0_paths': B0_path, 'B1_paths': B1_path, 'B2_paths': B2_path}




    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
