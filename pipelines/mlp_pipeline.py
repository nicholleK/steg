import os, sys
import pickle

import numpy as np # for arrays
import cv2 # image loading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from pipelines.pipeline import Pipeline
from models.manual_mlp import ManualMLP
from preprocessing.srm_filters import SRMFilter
from features.feature_extractor import FeatureVector
from results.classification_result import ClassificationResult

# constants
CLASSES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}


class MLP_Pipeline(Pipeline):


    def __init__(self):
        super().__init__(name = "HPF + MLP")

        self.srm = SRMFilter() # srm filter instance
        self.scaler = None # feature normaliser
        self.report = None # store metrics
        self.input_size = None  # size of feature vector
        self.n_classes = None # number of output classes
        self.lastAccuracy = 0.0 # accuracy from the most recent training run

        # Helpers


    def applyHPF(self, image: np.ndarray) -> np.ndarray:

        # apply srm hpf filters to an image and return a noise residual map

        return self.srm.extract_noise_residuals(image, normalise=False)

    # pipeline abstract methods implementattions

    def train(self, train_dir, val_dir = None, test_dir = None, epochs: int = 50, learning_rate:float = 0.001, batch_size:int =128, test_size:float = 0.2, random_state:int = 42, **kwargs) -> dict:

#          hpf + mlp train pipeline
#         loads image and extracts features, normalises features,
#         splits into train/test sets, train mlp and evaluate test sets

        print(f"\n____ {self.name} Pipeline: Training ____")
        self.status = "training"

#         Load dataset
        def get_dataset(data_dir, cache_path):

            if data_dir is None:
                return  None, None

            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_path}")
                data = np.load(cache_path)
                return  data["X"], data["Y"]

            X, Y = self._load_dataset(data_dir)
            np.savez(cache_path, X=X, Y=Y)
            print(f"Cached features to {cache_path}")
            return X,Y

        # load training data

        print(f"\nLoading training dataset...")


        X_train, Y_train = get_dataset(train_dir, cache_path="features_cache_train.npz")

        # train the RF
        print(f"Training dataset:"
              f"{X_train.shape[0]} samples | "
              f"{X_train.shape[0]} features per sample ")


        #  load validation data
        X_val = Y_val = None

        if val_dir is not None:
            print("\n Loading validation dataset...")

            X_val, Y_val = get_dataset(val_dir, cache_path="features_cache_val.npz")

            print(f"Validation dataset:"
                  f"{X_val.shape[0]} samples | "
                  f"{X_val.shape[0]} features per sample ")

    #      Load test data

        X_test = Y_test = None

        if test_dir is not None:
            print("\n Loading test dataset...")

            X_test, Y_test = get_dataset(test_dir, cache_path="features_cache_test.npz")

            print(f"Test dataset:"
                  f"{X_test.shape[0]} samples | "
                  f"{X_test.shape[0]} features per sample ")


#         normalise features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        if X_val is not None:
            X_val = self.scaler.fit_transform(X_val)

        if X_test is not None:
            X_test = self.scaler.fit_transform(X_test)


#       Initialise Model
        self.input_size = X_train.shape[1]
        self.n_classes = len(set(Y_train))

        print(
            f"\nInitializing MLP..."
            f"\nInput features: {self.input_size}"
            f"\nHiddem layer: 64"
            f"\nOutput classes: {self.n_classes}"
        )

        model = ManualMLP(
            layer_sizes=[
                self.input_size,
                64,
                self.n_classes
            ],
            learning_rate=learning_rate
        )

#       Training
        print(f"\n Training MLP...")

        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size
        )

        print(f"\nTraining complete.")

        # Validation eval

        val_accuracy = None
        val_cm = None
        val_report = None
        Y_val_pred = None
        Y_val_proba = None

        if X_val is not None:
            print(f"Evaluating on a validation set...")

            Y_val_pred = model.predict(X_val)
            Y_val_proba = model.predict(X_val)
            val_accuracy = accuracy_score(Y_val,Y_val_pred)
            val_cm = confusion_matrix(Y_val, Y_val_pred)
            present = sorted(set(Y_val) | set(Y_val_pred))
            val_report = classification_report(Y_val, Y_val_pred, labels=present,
                                       target_names=[CLASSES[i] for i in present],
                                       zero_division=0,
                                       output_dict=True
                                       )

            print(f"\n Validation Accuracy: {val_accuracy: 4f} ({val_accuracy * 100:.2f}%))")
            print(f"\n Validation Classification Report: \n{val_report}")
            print(f"\n Validation Confusion Matrix: \n{val_cm}")


        #  test eval


        test_accuracy = None
        test_cm = None
        test_report = None
        Y_test_pred = None
        Y_test_proba = None

        if X_test is not None:
            print(f"Evaluating on a test set...")

            Y_test_pred = model.predict(X_test)
            Y_probability = model.predict_proba(X_test)
            test_accuracy = accuracy_score(Y_test, Y_test_pred)
            test_cm = confusion_matrix(Y_test, Y_test_pred)
            present = sorted(set(Y_test) | set(Y_test_pred))

            test_report = classification_report(Y_test, Y_test_pred, labels=present,
                                                target_names=[CLASSES[i] for i in present],
                                                zero_division=0,
                                                output_dict=True
                                                )

            print(f"\n Test Accuracy: {test_accuracy: 4f} ({test_accuracy * 100:.2f}%))")
            print(f"\n Test Classification Report: \n{test_report}")
            print(f"\n Test Confusion Matrix: \n{test_cm}")


#             store trained model and scaler for classify() and saveModel()

        self.activeModel = model

        if test_accuracy is not None:
            self._last_accuracy = test_accuracy
            self.report = test_report
        elif val_accuracy is not None:
            self._last_accuracy = test_accuracy
            self.report = test_report
        else:
            self._last_accuracy = 0.0
            self.report = None
        self.status = "ready"

        return {
            "model":  model,
            "scaler": self.scaler,
            "train_samples": len(Y_train),

            "val_samples": (
                len(Y_val)
                if Y_val is not None
                else 0
            ),

            "test_samples": (
                len(Y_test)
                if Y_test is not None
                else 0
            ),
            "train_accuracy": None,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "validation_confusion_matrix": val_cm,
            "validation_classification_report": val_report,
            "test_confusion_matrix": test_cm,
            "test_classification_report": test_report,
            "loss_history": history["loss_history"],
            "X_test" : X_test,
            "X_val" : X_val,
            "Y_val" : Y_val,
            "Y_test" : Y_test,
            "Y_pred" : Y_test_pred,
            "Y_proba" : Y_test_proba,
            "Y_val_pred" : Y_val_pred,
            "Y_val_proba" : Y_val_proba,
        }

    def classify(self,image_path: str, threshold: float = 0.60) -> 'ClassificationResult':
#         classifies a single image using the model
#         loads image, apply filters, extract feature vector,
#         normalise features, forward pass through MLP and return ClassificationResult (predicted class, confidence, probabilities)

        if self.activeModel is None:
            raise RuntimeError("No active model.")
        if self.scaler is None:
            raise RuntimeError("No scaler found.")

#         Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

#         Apply hpf
        residual = self.applyHPF(img)

#        Extract features
        fv = FeatureVector()
        fv.compute(residual)
        features = fv.getFv()

#       Apply saved scaler
        features = self.scaler.transform(features.reshape(1, -1))

#       Classify with MLP
        probs = self.activeModel.predict_proba(features)[0]

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        all_probs = {
            CLASSES[i]: round(float(p), 4)
            for i , p in enumerate(probs)
            if i in CLASSES
        }

        return ClassificationResult(
            predictedClass= CLASSES[predicted_idx],
            confidenceScore=confidence,
            allProbabilities=all_probs,
            threshold=threshold,
        )

    def saveModel(self, file_path:str = "models/hpf_mlp_model.pkl"):

        # saves the mlp model weights and fitted scaler in one file

        if self.activeModel is None:
            raise RuntimeError("No active model.")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        bundle = {
            "weights": self.activeModel.weights,
            "biases": self.activeModel.biases,
            "layer_sizes": self.activeModel.layer_sizes,
            "learning_rate": self.activeModel.learning_rate,
            "input_size": self.input_size, # used to rebuild architecture
            "n_classes": self.n_classes,  # used to rebuild architecture
            "scaler": self.scaler, # for feature normalisation
            "report": self.report,
        }

        with open(file_path, "wb") as f:
            pickle.dump(bundle, f)

        print(f" HPF + MLP model bundle saved to {file_path}")

    def loadModel(self, path:str = "models/hpf_mlp_model.pkl"):

#       loads mlp model bundle from disk and reconstructs the model
#       unpickle the bundle, rebuild the mlp architecture, load saved weights into the architecture and restore the scaler

        if not os.path.exists(path):
            raise FileNotFoundError("No saved model found.")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

#         reconstruct the architecture
        self.input_size = bundle["input_size"]
        self.n_classes = bundle["n_classes"]
        self.report = bundle.get("report", None)

#       load saved weights into new architecture
        model = ManualMLP(
            layer_sizes=bundle["layer_sizes"],
            learning_rate=bundle["learning_rate"]
        )
        model.weights = bundle["weights"]
        model.biases = bundle["biases"]

        self.activeModel = model
        self.scaler = bundle["scaler"]
        self.status = "ready"

        print(f" HPF + MLP model loaded from {path}")

#       MLP methods

    def validateSoftmax(self, imagePath:str) -> float:
#         runs an image through the model and verifies the output probs sum to 1.0

        result = self.classify(imagePath)
        total = sum(result.allProbabilities.values())
        print(f" Softmax sum: {total:.6f}")
        return total




