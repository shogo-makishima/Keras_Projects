# fit a mask rcnn on the kangaroo dataset
import os
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


class Shogo_Makishima_Dataset(Dataset):
    def load_dataset(self, datasetPath, trainFolder="train", testFolder="test", is_train=True):
        self.add_class(source="dataset", class_id=1, class_name="shogo makishima")

        if (is_train): self.__loadTrainDataset(f"{datasetPath}\\{trainFolder}")
        else: self.__loadTestDataset(f"{datasetPath}\\{testFolder}")

        print(f"[DATASET][IMAGES][{'TRAIN' if is_train else 'TEST' }] LEN = {len(self.image_info)}")


    def __loadTrainDataset(self, dataset_train_dir):
        image_files = list(filter(lambda x: x if x[-len("jpg"):] == "jpg" else None, os.listdir(dataset_train_dir)))
        xml_files = list(filter(lambda x: x if x[-len("xml"):] == "xml" else None, os.listdir(dataset_train_dir)))

        for index in range(min([len(image_files), len(xml_files)])):
            img_path = f"{dataset_train_dir}\\{image_files[index]}"
            ann_path = f"{dataset_train_dir}\\{xml_files[index]}"
            image_id = image_files[index][:-4]

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)


    def __loadTestDataset(self, dataset_test_dir):
        image_files = list(filter(lambda x: x if x[-len("jpg"):] == "jpg" else None, os.listdir(dataset_test_dir)))
        xml_files = list(filter(lambda x: x if x[-len("xml"):] == "xml" else None, os.listdir(dataset_test_dir)))

        for index in range(min([len(image_files), len(xml_files)])):
            img_path = f"{dataset_test_dir}\\{image_files[index]}"
            ann_path = f"{dataset_test_dir}\\{xml_files[index]}"
            image_id = image_files[index][:-4]

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)


    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)

        masks = zeros([h, w, len(boxes)], dtype="uint8")
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index("shogo makishima"))

        return masks, asarray(class_ids, dtype="int32")

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

class Shogo_Makishima_Config(Config):
    NAME = "shogo_makishima_cfg"
    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 250

    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 256

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Shogo_Makishima_Predictor_Config(Config):
    NAME = "shogo_makishima_cfg"
    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 256

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



train_dataset = Shogo_Makishima_Dataset()
train_dataset.load_dataset("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized", is_train=True)
train_dataset.prepare()

test_dataset = Shogo_Makishima_Dataset()
test_dataset.load_dataset("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized", is_train=False)
test_dataset.prepare()

isTrain = False

if isTrain:
    config = Shogo_Makishima_Config()

    print("[MODEL][STATE][TRAINING]")
    model = MaskRCNN(mode="training", model_dir="I:\GitHub\Shogo-Makishima\Datasets\Models\Shogo-Makishima\\", config=config)
    model.load_weights("I:\GitHub\Shogo-Makishima\Datasets\Models\\mask_rcnn_coco.h5", by_name=True,  exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    model.train(train_dataset, test_dataset, learning_rate=config.LEARNING_RATE, epochs=5, layers="heads")
else:
    config = Shogo_Makishima_Predictor_Config()

    print("[MODEL][STATE][TEST]")
    model = MaskRCNN(mode="inference", model_dir="I:\GitHub\Shogo-Makishima\Datasets\Models\Shogo-Makishima\\", config=config)
    model.load_weights("I:\GitHub\Shogo-Makishima\Datasets\Models\Shogo-Makishima\shogo_makishima_cfg20200714T1540\\mask_rcnn_shogo_makishima_cfg_0005.h5", by_name=True)

    # evaluate model on train and test dataset
    train_mAP = evaluate_model(train_dataset, model, config)
    test_mAP = evaluate_model(test_dataset, model, config)
    print(f"[NETWORK][EVALUATE][TRAIN] {round(train_mAP, 3)}\n[NETWORK][EVALUATE][TEST] {round(test_mAP, 3)}")