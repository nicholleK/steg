
import numpy as np # for arrays
import sys, os, cv2, pickle # file ops, image loading, model serialization

from sklearn.ensemble import RandomForestClassifier # sklearn's RF model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# add the project to python's path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pipelines.pipeline import Pipeline
from results.classification_result import ClassificationResult
from features.feature_extractor import FeatureVector


# constants
CLASSES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}
CONFIDENCE_THRESHOLD = 0.6  # default flag threshold

class RF_Pipeline(Pipeline):

    def __init__(self):
        super().__init__(name="Random Forest") # set name
        self.nTrees = 100 # tree size
        self.report = None
        self._last_accuracy = 0.0 # last obtained accuracy score after a training run


    # Abstract methods implementations

    # training method
    def train(self, train_dir, val_dir = None, test_dir = None, random_state: int = 42, **kwargs) -> dict:

        n_estimators = kwargs.get("n_estimators", self.nTrees)
        print(f"\n==={self.name} Pipeline: Training ===")
        self.status = "training"

        # dataset loader

        def get_dataset(data_dir, cache_path):
            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_path}")
                data = np.load(cache_path)
                return data["X"], data["Y"]

            X, Y = self._load_dataset(data_dir)
            np.savez(cache_path, X=X, Y=Y)
            print(f"Cached features to {cache_path}")
            return X, Y

        # load training data
        print(f"\nLoading training dataset...")
        X_train, Y_train = get_dataset(train_dir, cache_path="features_cache_train.npz")

        # train the RF
        print(f"Training samples: {len(Y_train)}")

        #  load validation data
        X_val = Y_val = None

        if val_dir is not None:
            print("\n Loading validation dataset...")

            X_val, Y_val = get_dataset(val_dir, cache_path="features_cache_val.npz")

            print(f"Validation samples: {len(Y_val)}")

        #      Load test data

        X_test = Y_test = None

        if test_dir is not None:
            print("\n Loading test dataset...")

            X_test, Y_test = get_dataset(test_dir, cache_path="features_cache_test.npz")

            print(f"Test samples: {len(Y_test)}")

        # Train RF

        print(f"Training RF ({n_estimators} trees)...")

        classifier = RandomForestClassifier(n_estimators=n_estimators,
                                            max_features=0.3,
                                            n_jobs=-1,
                                            random_state=random_state)
        classifier.fit(X_train, Y_train) # training
        print(f"\nTraining complete.")

        # Validation eval

        print(f"Evaluating on a test set...")
        val_accuracy = None
        val_cm = None
        val_report = None

        if X_val is not None:
            print(f"Evaluating on a validation set...")

            Y_val_pred = classifier.predict(X_val)
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

        # Test eval

        test_accuracy = None
        test_cm = None
        test_report = None
        Y_test_pred = None
        Y_probability = None

        if X_test is not None:
            print(f"Evaluating on a test set...")

            Y_test_pred = classifier.predict(X_test)
            Y_probability = classifier.predict_proba(X_test)
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

        #  store the model

        # store the model and accuracy for evaluation
        self.activeModel = classifier

        if test_accuracy is not None:
            self._last_accuracy = test_accuracy
            self.report = test_report
        elif val_accuracy is not None:
            self._last_accuracy = test_accuracy
            self.report = test_report
        self.status = "ready"

        # return results

        return {
            "model": classifier,
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
            "X_test": X_test,
            "Y_test": Y_test,
            "Y_pred": Y_test_pred,
            "Y_probability": Y_probability,
            "feature_importance": classifier.feature_importances_,
        }


    # classify image using trained model
    def classify(self,image_path: str, threshold:float=0.60) -> ClassificationResult:
        if self.activeModel is None:
            raise RuntimeError("No model loaded")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        residual = self.srm.extract_noise_residuals(img, normalise=False) # apply filters

        # extract statistical features to create fv
        feature_vector = FeatureVector()
        feature_vector.compute(residual)

        features = feature_vector.getFv().reshape(1,-1) # convert fv into 2D row for classification
       # classify
        probability = self.activeModel.predict_proba(features)[0]
        predicted_idx = int(np.argmax(probability))
        confidence = float(probability[predicted_idx])

    # map class indices to names for allProbabilities dict
        all_probabilities ={
        CLASSES[i]: round(float(p), 4)
        for i, p in enumerate(probability)
        if i in CLASSES
        }
        
        return ClassificationResult(
            predictedClass=CLASSES[predicted_idx],
            confidenceScore=confidence,
            allProbabilities=all_probabilities,
            threshold=threshold,
        )


    def saveModel(self, file_path: str = "models/rf_model.pkl"):
        # serializes trained model to disk using pickle
        if self.activeModel is None:
            raise RuntimeError("No model to save.")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        bundle = {
            "model": self.activeModel,
            "report": self.report,
        }

        with open(file_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f" Model saved: {file_path}")

    def loadModel(self, file_path: str = "models/rf_model.pkl"):
        #load saved model from disk
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find model at: {file_path}") # raise if file doesn't exist

        with open(file_path, "rb") as f:
            bundle = pickle.load(f)

        self.activeModel = bundle["model"]
        self.report = bundle.get("report", None)
        self.status = "ready"
        print(f" Model loaded from {file_path}")


    # RF methods
    def getFeatureImportance(self) -> list:
        # returns the importance score of each feature as determined by RF
        if self.activeModel is None:
            raise RuntimeError("No model loaded")
        return self.activeModel.feature_importances_.tolist()



















