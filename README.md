# ships-detection
 ships-detection is a CV model for ships detection on satellite images.


## Installation

Use the terminal to install ships-detection model.

```bash
git clone https://github.com/BRomanchuk/ships-detection.git
```

## Project structure
```
.
├── EDA.ipynb               # Jupyter notebook with EDA of the dataset 
├── main.py                 # test of the UNet model 
├── train.py                # training file of the UNet model
├── model                   # folder with model files
│   ├── conv                # folder with convolution blocks of UNet
│   │   ├── __init__.py     # basic convolution blocks
│   │   ├── downconv.py     # encoder block of UNet
│   │   └── upconv.py       # decoder block of UNet
│   ├── encod               # module with run-length encoding functions
│   ├── weights             # folder with model weights
│   ├── unet.py             # architecture of the UNet model
│   ├── model_params.py     # parameters of the UNet model
│   └── dataset.py          # custom dataset class for images
├── test_photos             # folder with photos to test the model
│   └── result_masks        # folder with masks predicted by the model
├── requirements.txt 
└── README.md
```

