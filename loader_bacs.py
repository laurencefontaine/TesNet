import torch
import os
import json
import os
import random
import glob


import tifffile as tif
import numpy as np
from torch.utils.data import Dataset
from settings_emma import *

import torchvision.transforms.functional as tf
from torchvision.transforms.functional import crop
from torchvision import transforms as T



class EmmaDataset(Dataset):
    def __init__(self, file_list, transform=None, crop_size=256):
        """
        Instantiate BacteriaDataset class.

        Parameters
        ----------
        transform :     torchvision.transforms.transforms.Compose OR single transform
                        Pytorch transformations that needs to be applied to the images
                        Default = None
        file_list :     string list
                        Predetermined list of files to use for the dataset
                        override root_dir when not none
                        Default = None
        """
        self.transform = transform
        self.file_list = file_list
        self.crop_size = crop_size
        self.crops_list = self.crops_idx()


    def crops_idx(self):
        crop_list = []
        
        label = self.__get_label__(self.file_list)
        #print(self.file_list)
        for file in self.file_list:
            #np_img = torch.from_numpy(tif.imread(file).astype('float32'))
            split_path = os.path.split(file)
            #print(split_path)
            mask_path = os.path.join(split_path[0], 'Masks', 'MASK_{}'.format(split_path[1]))
            np_mask = torch.from_numpy(tif.imread(mask_path).astype('float32'))
            # print(mask_path)
            for j in range(0, np_mask.shape[-2], self.crop_size): 
                for i in range(0, np_mask.shape[-1], self.crop_size):
                    crop_index = (slice(j, j + self.crop_size), slice(i, i + self.crop_size))
                    crop_mask = np_mask[crop_index]
                    
                    if (np.count_nonzero(crop_mask) > 0.1 * self.crop_size * self.crop_size) and crop_mask.shape[-1] == self.crop_size:
                        #crop_idx = (slice(None, None), slice(0, 512), slice(0, 512))
                        #crop_img = np_img[crop_idx]
                        crop_list.append([file, j, i, label])



            
            #crop_function = T.TenCrop(512)
            #crops = crop_function(np_img)
            # for j in range(0, image.shape[-2], 1152): #itère en y
            #     for i in range(0, image.shape[-1], 1152): #itère en x
            #         crops = T.FiveCrops(512)

            # for crop in crops:
                
            #     crop_list.append([crop, label])

        return crop_list
                
    def __len__(self):
        return len(self.crops_list)
    
    
    def __getitem__(self, idx):
        file, j, i, label = self.crops_list[idx]
        np_img = torch.from_numpy(tif.imread(file).astype('float32'))
        crop_idx = (slice(None, None), slice(j, j + self.crop_size), slice(i, i + self.crop_size))
        tensor_img = np_img[crop_idx]
        #tensor_img = torch.from_numpy(crop.astype('float32'))

        # if ".nd2" in path:
        #     tensor_img = torch.from_numpy(nd2.imread(path).astype('float32'))
        # else:
        
        # if self.use_centroids:
        #     tensor_img = self.__crop_based_centroids__(path, tensor_img)

        if self.transform is not None:
            tensor_img = self.transform(tensor_img)

        return tensor_img, label
        
    def __get_label__(self, img_path):
        label = 0
        if "Cip" in img_path:
            label = 1
        elif "Fos" in img_path:
            label = 2
        elif "PmB" in img_path:
            label = 3

        return label

def split_files(path):
    random.seed(42)
    UT = []
    CIP = []
    FOS = []
    PMB = []

    for directory in glob.glob(path + '/**/*.tif', recursive=True):
            # if "9731" in directory:
            # if "9731_" in directory:
            #     UT.append(directory)
        if "Mask" not in directory and "9734" not in directory:
            if "Cip" in directory:
                CIP.append(directory)
            elif "Fos" in directory:
                FOS.append(directory)
            elif "PmB" in directory:
                PMB.append(directory)
            else:
                UT.append(directory)

    # for directory in glob.glob(path + '/**/*.nd2', recursive=True):
    #         if "9731" in directory:
    #             if "9731_" in directory:
    #                 UT.append(directory)
    #             if "Cip" in directory:
    #                 CIP.append(directory)
    #             elif "Fos" in directory:
    #                 FOS.append(directory)
    #             elif "PmB" in directory:
    #                 PMB.append(directory)


    random.shuffle(UT)
    random.shuffle(CIP)
    random.shuffle(FOS)
    random.shuffle(PMB)

    training_files = UT[:23] + CIP[:23] + FOS[:23] + PMB[:23]
    random.shuffle(training_files)
    valid_files = UT[23:27] + CIP[23:27] + FOS[23:27] + PMB[23:27]
    random.shuffle(valid_files)
    test_files = UT[27:31] + CIP[27:31] + FOS[27:31] + PMB[27:31]
    random.shuffle(test_files)

    return training_files, valid_files, test_files

# def main(path, channels):
#     # args = parser.parse_args()
#     # with open(args.json_path) as json_file:
#     #     parameters = load(json_file)

def crops_emma(dataset_dir):
    folders = os.listdir(dataset_dir)
    UT = []
    CIP = []
    FOS = []
    PMB = []
    for directory in glob.glob(dataset_dir + '/**/*.tif', recursive=True):
        #print(directory)
        if "Mask" not in directory and "9734" not in directory:
            if "Cip" in directory:
                CIP.append(directory)
            elif "Fos" in directory:
                FOS.append(directory)
            elif "PmB" in directory:
                PMB.append(directory)
            else:
                UT.append(directory)
        

    print(UT, CIP, FOS, PMB)
#print('allo1')

if __name__ == '__main__':
    path = 'datasets/EmmaOMDisruptors20230420_tif'
    #crops_emma(path)


    #in_channels = [0, 1, 2]
    #main(path, in_channels)

    #print(os.listdir(path))
    #print('allo')
    files = []
    for directory in glob.glob(path + '/**/*.tif', recursive=True):
        #print(directory)
        if "Mask" not in directory and "9734" not in directory:
            files.append(directory)
    #print(files)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    #print(files)
    for file in files:
        image = torch.Tensor(tif.imread(file).astype('float32'))

        # mean_tensor = torch.mean(image, dim=[1,2])
        # mean += mean_tensor

        std_mean = torch.std_mean(image, dim=[1,2])
        std += std_mean[0]
        mean += std_mean[1]
        # print(np.mean(image,))
        # for i, channel in enumerate(image):
        #     #print(mean[i])
        #     mean[i] += (torch.mean(channel))
        #     std[i] += np.std(channel)
            #print(np.mean(channel))
    #print(mean)
    print(mean, np.array(mean)/len(files))
    print(std, np.array(std)/len(files))
    # best_loss = 100000.0
    # training_files, validation_files, test_files = split_files(path)
    # validation_data = EmmaDataset(validation_files)
    # print(len(validation_data))

    # print(len(training_files))
    # tail = '8_D1.tif'
    # mask_path = os.path.join(path, '9731_P1', 'Masks', 'MASK_{}'.format(tail))

    # mask_img = torch.from_numpy(tif.imread(mask_path).astype('float32'))
    # print(mask_img.shape)
    # crop_index = (slice(0, 512), slice(0, 512))
    # crop = mask_img[crop_index]
    
    # print(crop.shape)
    # img_path = os.path.join(path, '9731_P1', tail)
    # img = torch.from_numpy(tif.imread(img_path).astype('float32'))

    # print(img.shape)
    # crop_index = (None, slice(0, 512), slice(0, 512))
    # crop_img = img[crop_index]
    # print(crop_img.shape)

    # training_data = EmmaDataset(transform=trans, training_files)
    # train_push_data = EmmaDataset(transform=None)
    

    # train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=train_batch_size)
    # valid_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=test_batch_size)
