import os
import numpy as np # for arrays
from abc import abstractmethod, ABC
from preprocessing.srm_filters import SRMFilter
from features.feature_extractor import FeatureVector

import cv2

from results.classification_result import ClassificationResult

# constants
CLASSES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}

def modelExists(file_path: str) -> bool:
    return os.path.exists(file_path)

# abstract base class for the pipelines
class Pipeline(ABC):
    def __init__(self,name):
        self.name = name # pipeline name
        self.status = "idle"
        self.activeModel = None # current active model
        self.srm = SRMFilter()  # SRM filter instance
# load, preprocess, train models
    @abstractmethod
    def train(self,data_dir: dict, **kwargs) -> dict:
        pass

# classify image using trained model
    @abstractmethod
    def classify(self,image_path: str) -> 'ClassificationResult':
        pass
# save trained model to disk
    @abstractmethod
    def saveModel(self,file_path: str):
        pass

# load trained model from disk
    @abstractmethod
    def loadModel(self,file_path: str):
        pass

    # load dataset
    def _load_dataset(self, data_dir:dict, **kwargs) -> tuple:
        X, Y = [],[]

        for label, dir in data_dir.items():
            #path validation
            if not os.path.exists(dir):
                print(f"Warning: {dir} does not exist")
                continue
            # get only image files with the supported formats
            files = [f for f in os.listdir(dir)
                     if f.lower().endswith(('.pgm', '.png', '.tiff', '.tif'))]

            # feedback mssg
            print(f"Loading class '{CLASSES.get(label, label)}': "
                  f"{len(files)} images from {dir}")

            for fname in files:
                img = cv2.imread(os.path.join(dir, fname), cv2.IMREAD_GRAYSCALE) # load images as grayscale
                # image validation, skip if unreadable
                if img is None:
                    print("Couldn't load image:", fname)
                    continue
                # apply filters to get the noise residual image
                residuals = self.srm.extract_noise_residuals(img, normalise=False)

                #extract statistical features to create fv
                feature_vector = FeatureVector()
                feature_vector.compute(residuals)

                if not feature_vector.validate(): #validate the fv, skip if invalid fv
                    print("Invalid feature:", fname)
                    continue

                X.append(feature_vector.getFv()) # append the fv
                Y.append(label)
                print(f"Finished classes {CLASSES[label]}: {len(X)} total samples")

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

    # Dataset loader for ViT and CNN
    def _load_dataset_residuals(self, data_dir:dict, img_size=64,max_per_class:int=None) -> tuple:
        X, Y = [],[]

        for label, dir in data_dir.items():
            #path validation
            if not os.path.exists(dir):
                print(f"Warning: {dir} does not exist")
                continue
            # get only image files with the supported formats
            files = sorted(
                [
                    f for f in os.listdir(dir)
                    if f.lower().endswith(('.pgm', '.png', '.tiff', '.tif'))
                ],
                key=lambda x: int(os.path.splitext(x)[0])
                if os.path.splitext(x)[0].isdigit()
                else x
            )


            if max_per_class is not None:
                files = files[:max_per_class]

            # feedback mssg
            print(f"Loading class '{CLASSES.get(label, label)}': "
                  f"{len(files)} images from {dir}")


            for fname in files:
                img = cv2.imread(os.path.join(dir, fname), cv2.IMREAD_GRAYSCALE) # load images as grayscale
                # image validation, skip if unreadable
                if img is None:
                    print("Couldn't load image:", fname)
                    continue
                # apply filters to get the noise residual image
                residuals = self.srm.extract_noise_residuals(img, normalise=False)

                #
                h,w = img.shape[:2]
                start_h = (h - img_size) // 2
                start_w = (w - img_size) // 2

                resized_channels = []
                # crop to fixed spatial size 4 ViT path embedding
                for res in residuals:
                    crop = res[start_h:start_h+img_size, start_w:start_w+img_size]
                    resized_channels.append(crop)

                stacked = np.stack(resized_channels, axis=0).astype(np.float32)
                stacked = np.clip(stacked, -3.0, 3.0) / 3.0 # scale to [-1,1]

                # standardize per channel
                X.append(stacked) # append the fv
                Y.append(label)

            print(f"Finished classes {CLASSES[label]}: {len(X)} total samples")

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

    # helper: returns the last accuracy score of loaded model if available
    def evaluate(self) -> float:
        #validation
        if self.activeModel is None:
            return 0.0
        return getattr(self, '_last_accuracy', 0.0)

    # helper: return true if trained model is loaded and ready
    def isReady(self) -> bool:
        return self.activeModel is not None

    # helper: validation checj for model existence on disk
    #  returns string rep of object
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, status={self.status})"

























