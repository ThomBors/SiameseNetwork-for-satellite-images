import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


def crate_TensorDataset_nomalized(image_list,label_list, pairs = 5,transform=None):
    """
    image_list --> array of immages
    label_list --> array of lable
    pairs --> number of pairs for una immage
    tranfrom --> tranformation applicable to tensor

    returen a tensor data with nomalized data
    """
    left_input = []
    right_input = []
    targets = []
    # normalization parameter acquisition
    
    imag_nomr = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    #Number of pairs per image
    pairs = pairs
    #Let's create the new dataset to train on
    for i in range(len(label_list)):
        for _ in range(pairs): 
            # compare the same immage on the left to different immages to the right
            compare_to = i 
            while compare_to == i: #Make sure it's not comparing to itself
                compare_to = random.randint(0,len(image_list)-1)

            # np.array to tenso
            left_img = image_list[i][0]
            left_img = np.transpose(left_img,(2,0,1))
            left_img = torch.Tensor(left_img)

            right_img = image_list[compare_to][0]
            right_img = np.transpose(right_img,(2,0,1))
            right_img = torch.Tensor(right_img)

            # nomalization of immages
            left_img = imag_nomr(left_img)
            right_img = imag_nomr(right_img)

            # create data sets
            left_input.append(np.array(left_img))
            right_input.append(np.array(right_img))

            if label_list[i] == label_list[compare_to]:# They are the same
                targets.append(1.)
            else:# Not the same
                targets.append(0.)

            # apply data augemntation tecniques
            if transform :
                left_img = transform(left_img)
                right_img = transform(right_img)
                
                left_input.append(np.array(left_img))
                right_input.append(np.array(right_img))

                # need to update also the data tagets
                if label_list[i] == label_list[compare_to]:# They are the same
                    targets.append(1.)
                else:# Not the same
                    targets.append(0.)
    
    left_input = np.squeeze(np.array(left_input))
    left_input = torch.Tensor(left_input)
    right_input = np.squeeze(np.array(right_input))
    right_input = torch.Tensor(right_input)

    targets = np.squeeze(np.array(targets))
    targets = torch.Tensor(targets)
    targets = targets.unsqueeze(1)
    
    dataset = TensorDataset(left_input,right_input,targets)
    
    return dataset
    




