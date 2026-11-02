import torch
import torchvision
from torchvision import transforms

def WoofLoader(batch_size = 32, shuffle = True, num_workers = 3):

    transform = transforms.Compose([
        transforms.CenterCrop(size=(160,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(root="/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/DiffusICE/examples/imagewoof/imagewoof2-160", transform = transform)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers), len(dataset)