import os
import torch
import torchvision

from dataset.wm811k import WM811K

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

def load_dataset(data_name='wm811k', path="./data/WM811K_labeled.pkl", train=True, val=False,
                 supcon=False, input_channels=1, img_size=(32, 32), num_classes=8, max_per_class=-1, 
                 list_classes=[0, 1, 2, 3, 4, 5, 6, 7], idx=None, apply_filter=False):
    if data_name == 'wm811k':
        if train or val:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(90),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, ) * input_channels, (0.5, ) * input_channels)
            ]) 
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, ) * input_channels, (0.5, ) * input_channels)
            ]) 
            
        if supcon:
            transforms = TwoCropTransform(transforms)
            
        dataset = WM811K(path=path, train=train, val=val, transforms=transforms, input_channels=input_channels, 
                         num_classes=num_classes, max_per_class=max_per_class, list_classes=list_classes, idx=idx,
                         apply_filter=apply_filter)

    
    return dataset


def get_loaders(train_dataset, test_dataset, val_dataset=None, batch_size=128):
    
    if train_dataset != None:
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   #sampler=sampler,
                                                   batch_size=batch_size,
                                                   shuffle=True, 
                                                   num_workers=2) 
    else:
        train_loader = None

    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
        test_loader = None
        
    if val_dataset != None:
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
    else:
        val_loader = None    

    return train_loader, test_loader, val_loader