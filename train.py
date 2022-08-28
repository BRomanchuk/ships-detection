import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import ImgDataset
from runlength import get_mask_encodings

from unet import UNet


class model_param:
    img_size = (80, 80)
    batch_size = 8
    num_workers = 4
    lr = 0.001
    epochs = 3
    unet_depth = 5
    unet_start_filters = 8
    log_interval = 70


def get_annos():
    annos = pd.read_csv(anno_fpath)
    annos['EncodedPixels_flag'] = annos['EncodedPixels'].map(lambda x: 1 if isinstance(x, str) else 0)

    imgs = annos.groupby('ImageId').agg({'EncodedPixels_flag': 'sum'}).reset_index().rename(
        columns={'EncodedPixels_flag': 'ships'})

    imgs_w_ships = imgs[imgs['ships'] > 0]
    imgs_wo_ships = imgs[imgs['ships'] == 0].sample(20000, random_state=69278)

    selected_imgs = pd.concat([imgs_w_ships, imgs_wo_ships])
    selected_imgs['has_ship'] = selected_imgs['ships'] > 0

    train_imgs, test_imgs = train_test_split(selected_imgs, test_size=0.15, stratify=selected_imgs['has_ship'],
                                             random_state=42)
    train_fnames = train_imgs['ImageId'].values
    test_fnames = test_imgs['ImageId'].values

    return  train_fnames, test_fnames, annos


def transform_data():
    train_transforms = transforms.Compose([
        transforms.Resize(model_param.img_size),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(model_param.img_size),
        transforms.ToTensor()
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(model_param.img_size)
    ])
    return train_transforms, test_transforms, mask_transforms


def load_datasets(params):
    train_dpath, train_fnames, test_fnames, annos, original_img_size = params
    train_transforms, test_transforms, mask_transforms = transform_data()

    train_ds = ImgDataset(train_dpath, train_fnames, train_transforms, get_mask_encodings(annos, train_fnames),
                          original_img_size, mask_transforms)
    test_ds = ImgDataset(train_dpath, test_fnames, test_transforms, get_mask_encodings(annos, test_fnames),
                         original_img_size, mask_transforms)

    return train_ds, test_ds


def get_data_and_model(train_ds, test_ds):
    train_dl = DataLoader(train_ds, batch_size=model_param.batch_size, num_workers=model_param.num_workers)
    test_dl = DataLoader(train_ds, batch_size=model_param.batch_size, num_workers=model_param.num_workers)

    model = UNet(2, depth=model_param.unet_depth, start_filters=model_param.unet_start_filters, merge_mode='concat')
    optim = torch.optim.Adam(model.parameters(), lr=model_param.lr)

    return train_dl, test_dl, model, optim


def get_loss(dl, model):
    loss = 0
    for X, y in dl:
        X, y = Variable(X).cuda(), Variable(y).cuda()
        output = model(X)
        loss += F.cross_entropy(output, y).item()
    loss = loss / len(dl)
    return loss


def train_model(train_dl, test_dl, model, optim):
    iters = []
    train_losses = []
    test_losses = []

    it = 0
    min_loss = np.inf

    model.train()
    for epoch in range(model_param.epochs):
        for i, (X, y) in enumerate(train_dl):
            X = Variable(X).cuda()  # [N, 1, H, W]
            y = Variable(y).cuda()  # [N, H, W] with class indices (0, 1)
            output = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i + 1) % model_param.log_interval == 0:
                it += model_param.log_interval * model_param.batch_size
                iters.append(it)
                train_losses.append(loss.item())

                model.eval()
                test_loss = get_loss(test_dl, model)
                model.train()
                test_losses.append(test_loss)
                print(f'Epoch {epoch}. Step {i}. Train loss {loss.item():.5f}. Test_loss {test_loss:.5f}')

    return iters, train_losses, test_losses


if __name__ == '__main__':
    train_dpath = 'airbus-ship-detection/train_v2/'
    anno_fpath = 'airbus-ship-detection/train_ship_segmentations_v2.csv'
    original_img_size = (768, 768)

    train_fnames, test_fnames, annos = get_annos()

    ds_params = (train_dpath, train_fnames, test_fnames, annos, original_img_size)

    train_ds, test_ds = load_datasets(ds_params)
    train_dl, test_dl, model, optim = get_data_and_model(train_ds, test_ds)

    iters, train_losses, test_losses = train_model(train_dl, test_dl, model, optim)
