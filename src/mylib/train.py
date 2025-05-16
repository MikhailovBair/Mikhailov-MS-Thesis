import numpy as np
import torch

from tqdm.auto import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

import os
import time
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-i", "--save_interval", type=int)
args = parser.parse_args()

writer = SummaryWriter()

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-60, +60)),
    v2.ToDtype(torch.float32, scale=True),
]
)


class EndocastSegmentationDataset2D(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = list(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask_ > 0

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


train_dataset = EndocastSegmentationDataset2D('./resized_train_dataset/img', './resized_train_dataset/ann', transforms)
val_dataset = EndocastSegmentationDataset2D('./resized_val_dataset/img', './resized_val_dataset/ann', transforms)
batch_size = args.batch_size
shuffle_dataset = True
random_seed = 239017

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

dataloaders = {
    'train': train_loader,
    'val': validation_loader
}


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def compute_metrics(prediction, target):
    mask = np.zeros_like(prediction)
    prediction = F.sigmoid(torch.tensor(prediction))
    mask[prediction >= 0.5] = 1
    inter = np.sum(mask * target, axis=tuple(range(1, len(mask.shape))))
    union = np.sum(mask, axis=tuple(range(1, len(mask.shape)))) + np.sum(target,
                                                                         axis=tuple(range(1, len(target.shape))))
    positives = np.sum(mask, axis=tuple(range(1, len(mask.shape))))
    truthes = np.sum(target, axis=tuple(range(1, len(mask.shape))))

    epsilon = 1e-9
    dice = np.sum(2 * inter / (union + epsilon))
    precision = np.sum(inter / (positives + epsilon))
    recall = np.sum(inter / (truthes + epsilon))
    return dice, precision, recall


def train_and_val(model, dataloaders, optimizer, scheduler, criterion, num_epochs=100):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(1, num_epochs + 1), leave=False):
        metrics = {"train": {"dice": 0.0, "precision": 0.0, "recall": 0.0, "loss": 0.0},
                   "val": {"dice": 0.0, "precision": 0.0, "recall": 0.0, "loss": 0.0}}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for sample in dataloaders[phase]:
                inputs = sample[0].to(device)
                masks = sample[1].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy()
                    y_true = masks.data.cpu().numpy()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    metrics[phase]["loss"] += loss.item()
                    dice, precision, recall = compute_metrics(y_pred, y_true)
                    metrics[phase]["dice"] += dice
                    metrics[phase]["precision"] += precision
                    metrics[phase]["recall"] += recall

            dataset_length = len(dataloaders[phase].dataset)
            writer.add_scalar(f"Loss/{phase}", metrics[phase]["loss"] / dataset_length, epoch)
            writer.add_scalar(f"Dice/{phase}", metrics[phase]["dice"] / dataset_length, epoch)
            writer.add_scalar(f"Precision/{phase}", metrics[phase]["precision"] / dataset_length, epoch)
            writer.add_scalar(f"Recall/{phase}", metrics[phase]["recall"] / dataset_length, epoch)

        if scheduler:
            scheduler.step()

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, '/gpfs/gpfs0/bair.mikhailov/crocodile_experiments/checkpoints/model_weight{}'.format(epoch + 300))

    return model


epochs = args.num_epochs


def train():
    # global trained_model
    model = UNet(in_channels=1, n_class=1)
    checkpoint = torch.load("/gpfs/gpfs0/bair.mikhailov/crocodile_experiments/checkpoints/model_weight300",
                            weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criterion = DiceBCELoss()
    trained_model = train_and_val(model, dataloaders,
                                  optimizer, scheduler,
                                  criterion, num_epochs=epochs,
                                  )
    return trained_model


trained_model = train()