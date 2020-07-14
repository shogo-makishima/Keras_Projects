from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import cv2, skimage.draw, numpy

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (numpy.sum(mask, -1, keepdims=True) >= 1)
        splash = numpy.where(mask, image, gray).astype(numpy.uint8)
    else:
        splash = gray.astype(numpy.uint8)
    return splash

class Shogo_Makishima_Predictor_Config(Config):
    NAME = "shogo_makishima_cfg"
    NUM_CLASSES = 1 + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = Shogo_Makishima_Predictor_Config()

print("[MODEL][STATE][TEST]")
model = MaskRCNN(mode="inference", model_dir="I:\GitHub\Shogo-Makishima\Datasets\Models\Shogo-Makishima\\", config=config)
model.load_weights("I:\GitHub\Shogo-Makishima\Datasets\Models\Shogo-Makishima\shogo_makishima_cfg20200714T1540\\mask_rcnn_shogo_makishima_cfg_0005.h5", by_name=True)

# Video capture
capture = cv2.VideoCapture(0)
SIZE = (128, 128)

success = True
while (success):
    # Read next image
    success, image = capture.read()
    if (success):
        resized = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Detect objects
        result = model.detect([rgb], verbose=0)[0]
        splash = color_splash(rgb, result['masks'])
        splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)

        cv2.imshow("Result", splash)
        cv2.waitKey(25)
        # Color splash
        # splash = color_splash(image, result['masks'])

# evaluate model on train and test dataset
# train_mAP = evaluate_model(train_dataset, model, config)
# test_mAP = evaluate_model(test_dataset, model, config)
# print(f"[NETWORK][EVALUATE][TRAIN] {round(train_mAP, 3)}\n[NETWORK][EVALUATE][TEST] {round(test_mAP, 3)}")
