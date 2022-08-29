import numpy as np


def get_mask_encodings(annos, fnames):
    """
    Collects the encodings of all ships for each image
    :param annos: pd.DataFrame with image annotations
    :param fnames: list of names of images
    :return: pd.DataFrame of encodings of all ships in the image
    """
    a = annos[annos['ImageId'].isin(fnames)]
    return a.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.tolist()).to_dict()


def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decodes run-length encodings into image mask
    :param mask_rle: string with run-length code
    :param shape: shape of the image
    :return: image mask
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    im = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        im[lo:hi] = 1
    return im.reshape(shape).T


def rle_encode(im):
    """
    Encodes image mask into run-length encodings
    :param im: image mask
    :return: run-length encoding of the mask
    """
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1
    return ' '.join(str(x) for x in runs)