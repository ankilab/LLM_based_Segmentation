import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import SegmentationDataset
from model import UNet
from sklearn.modelselection import traintest_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob


def splitdata(imagedir, maskdir, valsize=0.1, testsize=0.1, randomseed=42):
    imagepaths = glob.glob(os.path.join(imagedir, '*.png'))
    trainfiles, tempfiles = traintestsplit(imagepaths, testsize=valsize + testsize, randomstate=randomseed)
    valfiles, testfiles = traintestsplit(tempfiles, testsize=testsize / (valsize + testsize), randomstate=random_seed)

    return trainfiles, valfiles, test_files


def getdataloaders(imagedir, maskdir, batchsize=4, valsize=0.1, testsize=0.1):
    trainfiles, valfiles, testfiles = splitdata(imagedir, maskdir, valsize, testsize)

    def pathstoloader(image_paths):
        dataset = SegmentationDataset(imagedir, maskdir)

    subset = [Image.open(p) for p in image_paths]
    dataset.imagepaths, dataset.maskpaths = zip(*[(p, p.replace(imagedir, maskdir)) for p in image_paths])
    return DataLoader(dataset, batchsize=batchsize, shuffle=True)

    trainloader = pathstoloader(trainfiles)
    valloader = pathstoloader(valfiles)
    testloader = pathstoloader(testfiles)

    return trainloader, valloader, test_loader


def trainmodel(model, trainloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() * images.size(0)
    epochloss = runningloss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{numepochs}, Loss: {epochloss:.4f}')


def validatemodel(model, valloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.cuda(), masks.cuda()
    outputs = model(images)
    loss = criterion(outputs, masks)
    running_loss += loss.item() * images.size(0)
    valloss = runningloss / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss


def testmodel(model, testloader):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.cuda(), masks.cuda()
    outputs = model(images)
    yield images.cpu(), masks.cpu(), outputs.cpu()