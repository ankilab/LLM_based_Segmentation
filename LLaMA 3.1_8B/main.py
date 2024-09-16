from Dataset import SegmentationDataset
from model import UNet
from train import trainmodel, validatemodel, testmodel
from train import getdataloaders

if name == "main":

# Hyperparameters and paths

    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'
    batch_size = 4
    num_epochs = 25
    learning_rate = 0.001

    # DataLoader

    trainloader, valloader, testloader = getdataloaders(imagedir, maskdir, batchsize)

    # Model, Loss, Optimizer

    model = UNet().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Validation

    trainmodel(model, trainloader, criterion, optimizer, numepochs)
    validatemodel(model, val_loader, criterion)

    # Save model

    torch.save(model.statedict(), 'unetmodel.pth')

    # Testing and Visualization

    for images, masks, outputs in testmodel(model, test_loader):
        visualize_predictions(images, masks, outputs)