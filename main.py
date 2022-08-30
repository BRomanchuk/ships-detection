import torch
from torchvision.transforms import transforms

from PIL import Image

from model.unet import UNet

from model.model_params import model_params

if __name__ == '__main__':
    # create a model and load the weights
    model = UNet(2, depth=model_params.unet_depth, start_filters=model_params.unet_start_filters, merge_mode='concat')
    model.load_state_dict(torch.load('weights/model_weights.pt'))
    model.eval()

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # path to the test photo
    test_path = './test_photos/test_photo.jpg'

    # open PIL test image
    img = Image.open(test_path)

    # convert the PIL image to Torch tensor
    img_tensor = transform(img).float().cuda()
    img_tensor = img_tensor[None, :]

    # predicted Torch tensor
    predicted_tensor = model(img_tensor).cpu().detach()[0]

    # transform tensor into image and save the mask
    pred_mask = transforms.ToPILImage()(predicted_tensor).convert('RGB')
    pred_mask.save('./test_photos/result_masks/pred_mask.jpg')

