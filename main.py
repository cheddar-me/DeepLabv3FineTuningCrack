from pathlib import Path

import numpy as np
import click
import torch
from sklearn.metrics import accuracy_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model

def calculate_iou(preds, targets, num_classes):
    iou_list = []
    for i in range(1, num_classes):  # Start from 1 to exclude the background class
        intersection = torch.logical_and(preds == i, targets == i).sum().item()
        union = torch.logical_or(preds == i, targets == i).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        iou_list.append(iou)
    return np.nanmean(iou_list)  # Mean IoU, excluding background class




@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=15,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--num-classes",
              default=3,
              type=int,
              help="Specify the number of classes in the segmentation task.")
def main(data_directory, exp_directory, epochs, batch_size, num_classes):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

   # Specify the loss function for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    # Define metrics: accuracy and mean IoU
    metrics = {
        'accuracy': accuracy_score#,
        #'mean_iou': lambda preds, targets: calculate_iou(preds, targets, num_classes)
    }


    # Create the dataloader
    #dataloaders = datahandler.get_dataloader_single_folder(
    dataloaders = datahandler.get_dataloader_sep_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()
