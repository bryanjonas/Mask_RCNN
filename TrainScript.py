from mrcnn.sats_dataset import SatsDataset
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
import os
from glob import glob

import warnings
import numpy as np

#Root dir for saving stuff
ROOT_DIR = os.path.abspath("/lfs/jonas/maskrcnn/")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#Supress warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#Paths for training and validation
image_path = '/data/spacenet/bldg/data/train/MUL/'
geojson_path = '/data/spacenet/bldg/data/train/geojson/'
valid_image_path = '/data/spacenet/bldg/data/validation/MUL/'
valid_geojson_path = '/data/spacenet/bldg/data/validation/geojson/'

image_glob = glob(image_path + '*.tif')
valid_glob = glob(valid_image_path + '*.tif')

train_size = len(image_glob)
valid_size = len(valid_glob)

#Set training config (image size, etc)
class SatsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sats"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 48

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = train_size // IMAGES_PER_GPU

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = valid_size // IMAGES_PER_GPU
    
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (128,128)
    
config = SatsConfig()

# Training dataset
dataset_train = SatsDataset()
dataset_train.load_sats(image_path, geojson_path)
dataset_train.prepare()

# Validation dataset
dataset_val = SatsDataset()
dataset_val.load_sats(valid_image_path, valid_geojson_path)
dataset_val.prepare()

#Load the model
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

#Either load the COCO weights
model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
#***OR***
#Load the last weights from training
#model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20,
            verbose=1,
            layers='heads',
            max_queue=8, 
            workers=4)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10, 
            verbose=1,
            layers="all",
            max_queue=8, 
            workers=4)