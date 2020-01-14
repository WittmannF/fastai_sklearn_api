from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import sys
import os
import json

import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F



def get_trainLoader(batch_size, data_dir):
    print("get train loader")
    num_workers = 0
    
    
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    test_dir =  os.path.join(data_dir,'test')
    
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])


    data_transforms = {'train': transforms.Compose([transforms.Resize(size=224),
                                    transforms.CenterCrop((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         standard_normalization]),
                       'val': transforms.Compose([transforms.Resize(size=224),
                                    transforms.CenterCrop((224,224)),
                                         transforms.ToTensor(),
                                         standard_normalization]),
                       'test': transforms.Compose([transforms.Resize(size=224),
                                    transforms.CenterCrop((224,224)),
                                         transforms.ToTensor(), 
                                         standard_normalization])
                      }
    
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=False)
    data_loader = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    
    return data_loader
    


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda,last_validation_loss=None):
    """returns trained model"""
    if os.path.exists('model_transfer.pt'):
        model.load_state_dict(torch.load('model_transfer.pt'))
    
    
    def save_model(model, model_dir):
        path = os.path.join(model_dir, 'model_transfer.pt')
        torch.save(model.state_dict(), path)
    
    
    
    # initialize tracker for minimum validation loss
    
    if last_validation_loss is not None:
        valid_loss_min = last_validation_loss
    else:
        valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
            
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            save_model(model, args.model_dir)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
            
    # return trained model
    return model


if __name__ == '__main__':
    ## TODO: Specify model architecture 
    
    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    
    
    
    args = parser.parse_args()
    
    
    
    
    use_cuda = True
    
    model_transfer = models.resnet50(pretrained=True)

    for child in model_transfer.children():
        for param in child.parameters():
            param.requires_grad = False

    model_transfer.fc = nn.Linear(2048, 133)
    torch.nn.init.xavier_uniform(model_transfer.fc.weight)
    
    if use_cuda:
        model_transfer = model_transfer.cuda()
    
    criterion_transfer = nn.CrossEntropyLoss()
    optimizer_transfer = optim.SGD(filter(lambda p: p.requires_grad,model_transfer.parameters()), lr = 0.001)

    
    
    data_loader = get_trainLoader(args.batch_size, args.data_dir)
    
    
    
    
    
    train(15, data_loader, model_transfer, optimizer_transfer, criterion_transfer, use_cuda)