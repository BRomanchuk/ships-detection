import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import ImgDataset
from encodings.runlength import get_mask_encodings

from model_params import model_params

from unet import UNet


# get the names of train and test files, and the DataFrame with annotations
def get_annos():
    # read csv file with annotations, and create ship existence indicator column
    annos = pd.read_csv(anno_fpath)
    annos['EncodedPixels_flag'] = annos['EncodedPixels'].map(lambda x: 1 if isinstance(x, str) else 0)

    # group rows by ImageId and sum the number of ships on each image
    imgs = annos.groupby('ImageId').agg({'EncodedPixels_flag': 'sum'}).reset_index().rename(
        columns={'EncodedPixels_flag': 'ships'})

    # get images with and without ships
    imgs_w_ships = imgs[imgs['ships'] > 0]
    imgs_wo_ships = imgs[imgs['ships'] == 0].sample(20000, random_state=69278)

    # concatenate datasets with and without ships and create indicator column
    selected_imgs = pd.concat([imgs_w_ships, imgs_wo_ships])
    selected_imgs['has_ship'] = selected_imgs['ships'] > 0

    # split training and test images
    train_imgs, test_imgs = train_test_split(selected_imgs, test_size=0.15, stratify=selected_imgs['has_ship'],
                                             random_state=42)
    # get the names of training and test images
    train_fnames = train_imgs['ImageId'].values
    test_fnames = test_imgs['ImageId'].values

    return train_fnames, test_fnames, annos


# define transforms for images and their masks
def transform_data():
    train_transforms = transforms.Compose([
        transforms.Resize(model_params.img_size),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(model_params.img_size),
        transforms.ToTensor()
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(model_params.img_size)
    ])
    return train_transforms, test_transforms, mask_transforms


# create training and test data-loaders
def load_datasets(params):
    trn_dpath, trn_fnames, tst_fnames, anns, orgnl_img_size = params
    trn_transforms, tst_transforms, mask_transforms = transform_data()

    trn_ds = ImgDataset(trn_dpath, trn_fnames, trn_transforms, get_mask_encodings(anns, trn_fnames),
                          orgnl_img_size, mask_transforms)
    tst_ds = ImgDataset(trn_dpath, tst_fnames, tst_transforms, get_mask_encodings(anns, tst_fnames),
                         orgnl_img_size, mask_transforms)
    trn_dl = DataLoader(trn_ds, batch_size=model_params.batch_size, num_workers=model_params.num_workers)
    tst_dl = DataLoader(tst_ds, batch_size=model_params.batch_size, num_workers=model_params.num_workers)

    return trn_dl, tst_dl


# init model and optimizer
def get_model():
    mdl = UNet(2, depth=model_params.unet_depth, start_filters=model_params.unet_start_filters, merge_mode='concat')
    opt = torch.optim.Adam(model.parameters(), lr=model_params.lr)
    return mdl, opt


# calculate average loss
def get_loss(dl, mdl):
    loss = 0
    for X, y in dl:
        X, y = Variable(X).cuda(), Variable(y).cuda()
        output = mdl(X)
        loss += F.cross_entropy(output, y).item()
    loss = loss / len(dl)
    return loss


def train_model(trn_dl, tst_dl, mdl, opt):
    trn_losses = []  # list of train losses
    tst_losses = []  # list of test losses

    mdl.train()  # turn on training mode of the model
    for epoch in range(model_params.epochs):
        for i, (X, y) in enumerate(trn_dl):
            X = Variable(X).cuda()  # (N, 1, H, W) -- batch of images
            y = Variable(y).cuda()  # (N, H, W) -- batch of masks with class indices (0, 1)
            output = mdl(X)  # (N, 2, H, W) -- predicted batch of masks
            loss = F.cross_entropy(output, y)  # calculate cross-entropy loss

            opt.zero_grad()  # zero the gradient
            loss.backward()  # calculate gradient
            opt.step()  # backpropagation step

            if (i + 1) % model_params.log_interval == 0:
                mdl.eval()  # turn on evaluation mode
                test_loss = get_loss(tst_dl, mdl)  # calculate test loss

                # save model weights if current test loss is less than previous one
                if len(tst_losses) > 0 and test_loss < tst_losses[-1]:
                    torch.save(model.state_dict(), 'weights/model_weights.pt')

                tst_losses.append(test_loss)  # append test loss
                mdl.train()  # turn on training mode

                print(f'Epoch {epoch}. Step {i}. Train loss {loss.item():.5f}. Test_loss {test_loss:.5f}')
    return trn_losses, tst_losses


if __name__ == '__main__':
    # paths to the training data and annotations for masks
    train_dpath = 'airbus-ship-detection/train_v2/'
    anno_fpath = 'airbus-ship-detection/train_ship_segmentations_v2.csv'
    original_img_size = (768, 768)

    # get the filenames for training and test images
    train_fnames, test_fnames, annos = get_annos()

    # load train and test datasets
    ds_params = (train_dpath, train_fnames, test_fnames, annos, original_img_size)
    train_dl, test_dl = load_datasets(ds_params)

    # create model and optimizer
    model, optim = get_model()

    # train model and get train and test losses
    train_losses, test_losses = train_model(train_dl, test_dl, model, optim)
