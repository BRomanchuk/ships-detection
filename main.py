import torch

from unet import UNet

from model_params import model_params

if __name__ == '__main__':
    # create a model and load the weights
    model = UNet(2, depth=model_params.unet_depth, start_filters=model_params.unet_start_filters, merge_mode='concat')
    model.load_state_dict(torch.load('weights/model_weights.pt'))
    model.eval()
    # TODO test loader and saving the predicted mask
