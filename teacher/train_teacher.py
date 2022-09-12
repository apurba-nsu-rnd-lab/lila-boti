import os
from PIL import ImageFile
from PIL import Image
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

from models import Conv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train the model", default=10)
    parser.add_argument("-l", "--learning-rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument("-w", "--workers", type=int, help="Number of workers to train the model", default=4)
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size", default=20)
    parser.add_argument("-d", "--data-dir", type=str, help="Data directory containing {train, val, test} folders", default="sample_dataset")
    parser.add_argument("-s", "--save-dir", type=str, help="Model save directory", default="out")
    parser.add_argument("-n", "--save-model-name", type=str, help="Model save name", default="model.pt")
    return parser.parse_args()

def get_datasets(data_dir):
    train_dir = os.path.join(data_dir, 'train/')
    valid_dir = os.path.join(data_dir, 'val/')
    
    data_transforms = transforms.Compose([transforms.Resize(32),
                                      transforms.CenterCrop(32),
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485],
                                      std=[0.229])])

    image_datasets = {
        'train' : datasets.ImageFolder(root=train_dir,transform=data_transforms),
        'val' : datasets.ImageFolder(root=valid_dir,transform=data_transforms)
    }
    
    return image_datasets
    
def get_dataloaders(image_datasets):
    image_loaders = {
        'train' : torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size = args.batch_size, 
            num_workers = args.workers, 
            shuffle=True),
        'val' : torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size = args.batch_size, 
            num_workers = args.workers),
    }
    
    return image_loaders

def train(n_epochs, loaders, model, optimizer, criterion, device, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.
        total = 0.
        correctvl = 0.
        totalvl = 0.
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)

            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['val']):
            data, target = data.cuda(device), target.cuda(device)

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            
            # convert output probabilities to predicted class
            predvl = output.data.max(1, keepdim=True)[1]
            pre = predvl.data.cpu().argmax()
            # compare predictions to true label
            correctvl += np.sum(np.squeeze(predvl.eq(target.data.view_as(predvl))).cpu().numpy())
            totalvl += data.size(0)

        train_loss = train_loss/len(loaders['train'].dataset) ###
        valid_loss = valid_loss/len(loaders['val'].dataset) ###
        
        print('\nTrain Accuracy: %2d%% (%2d/%2d) \tvalid Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total, 100. * correctvl / totalvl, correctvl, totalvl))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        
        # Save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
    

if __name__ == "__main__":
    args = parse_cli_arguments()
    
    image_datasets = get_datasets(args.data_dir)
    image_loaders = get_dataloaders(image_datasets)
    
    class_names = image_datasets['train'].classes
    n_classes = len(class_names)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Train dataset size: {dataset_sizes['train']}, Validation dataset size: {dataset_sizes['val']}")
    
    class_names = image_datasets['train'].classes
    n_classes = len(class_names)
    print(f"Available classes in training set [{n_classes}]: {class_names}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = Conv2(n_classes)
    model = model.to(device)

    criterion_transfer = nn.CrossEntropyLoss()
    optimizer_transfer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    save_path = os.path.join(args.save_dir, args.save_model_name)
    model = train(args.epochs, image_loaders, model, optimizer_transfer, criterion_transfer, device, save_path)