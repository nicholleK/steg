import numpy as np # for arrays
import cv2 # image loading
import os, sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from pipelines.pipeline import Pipeline
from results.classification_result import ClassificationResult

# Constants
CLASSES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}

CONFIDENCE_THRESHOLD = 0.6

class ManualCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

#         Block 2

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Block 3

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

#         Block 4

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

#         Global Avg Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

#         Classification head. 256 - 128 - 4 classes

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,4),
        )

    def forward(self, x):
        # x shape: (batch_size, 12, 64, 64)

        # Map the 12 channels to 3 channels
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x

class CNN_Pipeline(Pipeline):

    def __init__(self):
        super().__init__(name = "CNN")

        self.epochs = 30
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.img_size = 64
        self.n_classes = 4
        self.num_channels = 12

        self.report = None
        self._last_accuracy = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"CNN device: {self.device}")

#         TRAIN

    def train(
            self,
            train_dir,
            val_dir = None, test_dir = None,
            random_state=42,
            **kwargs
    ) -> dict :

        print("\n-------CNN Pipeline: Training...-------")

        self.status = "training"

        # Training parameters

        epochs = kwargs.get("epochs", self.epochs)
        batch_size = kwargs.get("batch_size", self.batch_size)
        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        max_per_class = kwargs.get("max_per_class", None)

        # Load dataset
        print("\n-------Loading dataset...-------")

        def get_dataset(data_dir, cache_path):

            if data_dir is None:
                return  None, None

            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_path}")
                data = np.load(cache_path)
                return  data["X"], data["Y"]

            X, Y = self._load_dataset_residuals(data_dir,img_size = self.img_size, max_per_class=max_per_class)
            np.savez(cache_path, X=X, Y=Y)
            print(f"Cached features to {cache_path}")
            return X,Y

        # load training dataset

        print(f"\nLoading training dataset...")

        X_train, Y_train = get_dataset(train_dir, cache_path="cnn_cache_train.npz")

        if X_train is None or len(X_train) == 0:
            self.status = "idle"
            raise ValueError("No valid training samples were loaded")

        print(f"\n Training Dataset shape: {X_train.shape}")
        print(f"\n Training Labels shape: {Y_train.shape}")

        # validate training data shape

        if X_train.ndim != 4:
            self.status = "idle"
            raise ValueError(
                f"Expected X to have 4 dims "
                f"(samples, channels, height, width), "
                f"but received shape {X_train.shape} "
            )

        if X_train.shape[1] != self.num_channels:
            self.status = "idle"
            raise ValueError(
                f"Expected {self.num_channels} srm channels,"
                f"but received {X_train.shape[1]}"
            )

        # Load validation dataset

        X_val = Y_val = None

        if val_dir is not None:
            print("\n Loading validation dataset...")

            X_val, Y_val = get_dataset(val_dir, cache_path="cnn_cache_val.npz")

            print(f"Validation dataset shape{X_val.shape}  ")
            print(f"Validation labels shape{Y_val.shape} ")

        #      Load test data

        X_test = Y_test = None

        if test_dir is not None:
            print("\n Loading test dataset...")

            X_test, Y_test = get_dataset(test_dir, cache_path="cnn_cache_test.npz")

            print(f"\nTest dataset shape{X_test.shape}  ")
            print(f"Test labels shape{Y_test.shape} ")

        #        # convert arrays to pytorch tensors
        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).long()

        #         create dataloaders
        train_ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Validation Dataloader

        val_dl = None

        if X_val is not None:
            X_val_t = torch.from_numpy(X_val).float()
            Y_val_t = torch.from_numpy(Y_val).long()

            val_ds = TensorDataset(X_val_t, Y_val_t)
            val_dl = DataLoader(val_ds, batch_size=batch_size,shuffle=False)

        # Test Dataloader

        test_dl = None

        if X_test is not None:
            X_test_t = torch.from_numpy(X_test).float()
            Y_test_t = torch.from_numpy(Y_test).long()

            test_ds = TensorDataset(X_test_t, Y_test_t)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        #     Create CNN

        print("\nCreating manual CNN...")

        classifier = ManualCNN().to(self.device)

        print(classifier)

        # Loss function

        criterion = nn.CrossEntropyLoss()

        #         define optimization
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay)

        #         training loop
        print(f"\nTraining for {epochs} epochs...")
        loss_history = []

        for epoch in range(epochs):

            classifier.train()

            correct = 0
            total_loss = 0.0
            running_loss = 0.0

            # loop through each batch
            for batch_X, batch_Y in train_dl:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(
                    self.device)  # move the batches to the selected device

                #                  clear prev batch's grads
                optimizer.zero_grad()
                #                 forward pass
                logits = classifier(batch_X)
                #                 compute loss
                loss = criterion(logits, batch_Y)
                #                 backpropagation - grad calculation
                loss.backward()
                #                update weights
                optimizer.step()

                # Track loss
                running_loss += (loss.item() * batch_X.size(0))

            #     predictions
                predictions = logits.argmax(dim=1)

                correct += (predictions == batch_Y).sum().item()

                total_loss += batch_Y.size(0)

            epoch_loss = (running_loss / total_loss)
            epoch_accuracy = (correct / total_loss)

            # Validation during trainin

            val_epoch_accuracy = None

            if val_dl is not None:
                classifier.eval()

                val_correct = 0
                val_total = 0

                with torch.no_grad():  # disable grad calcs
                    # loop through each batch
                    for batch_X, batch_Y in val_dl:
                        batch_X = batch_X.to(self.device)  # move batch to selected device
                        logits = classifier(batch_X)  # run forward pass - run the model
                        preds = torch.argmax(logits,
                                             dim=-1).cpu().numpy()  # find class with largest logit, move it back to cpu and convert to numpy array
                        val_correct += (preds == batch_Y).sum().item()
                        val_total += batch_Y.size(0)

                val_epoch_accuracy = (val_correct / val_total)

            loss_history.append({
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "val_accuracy": val_epoch_accuracy
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                if val_epoch_accuracy is not None:
                    print(f" Epoch [{epoch + 1}/{epochs}] "
                          f" Loss: {epoch_loss:.4f} | "
                          f" Accuracy: {epoch_accuracy:.4f}"
                          f" Val Accuracy: {val_epoch_accuracy:.4f}"
                          )
                else:
                    print(f" Epoch [{epoch + 1}/{epochs}] "
                          f" Loss: {epoch_loss:.4f} | "
                          f" Accuracy: {epoch_accuracy:.4f}"
                          )

        def evaluate_dataset(self,model, dataloader):

            if dataloader is None:
                return None

            model.eval()

            all_preds = []  # predictions
            all_targets = []  # true labels
            all_probabilities = []

            with torch.no_grad():  # disable grad calcs
                # loop through each batch
                for batch_X, batch_Y in dataloader:

                    batch_X = batch_X.to(self.device)  # move batch to selected device
                    logits = model(batch_X)    # run forward pass - run the model
                    probs = torch.softmax(logits,dim=-1).cpu().numpy()
                    preds = torch.argmax(logits,dim=-1).cpu().numpy()  # find class with largest logit, move it back to cpu and convert to numpy array
                    all_preds.extend(preds.tolist())  # add to all predictions
                    all_targets.extend(batch_Y.numpy().tolist())  # add to all true labels
                    all_probabilities.extend(probs.tolist())

            all_targets = np.array(all_targets)
            all_probabilities = np.array(all_probabilities)
            all_preds = np.array(all_preds)
            accuracy = accuracy_score(all_targets, all_preds)
            cm = confusion_matrix(all_targets, all_preds, labels=list(CLASSES.keys()))
            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    all_targets,
                    all_preds,
                    labels=list(CLASSES.keys()),
                    average='weighted',
                    zero_division=0
                )
            )

            present = sorted(set(all_targets) | set(all_preds))
            report = classification_report(all_targets, all_preds, labels=list(CLASSES.keys()),
                                           target_names=[CLASSES[i] for i in present],
                                           zero_division=0, output_dict=True
                                           )

            # Print results

            print(f"\n Test Accuracy: {accuracy: 4f} ({accuracy * 100:.2f}%))")
            print(f"\n Classification Report: \n{report}")
            print(f"\n Confusion Matrix: \n{cm}")
            print(f"\n Weighted Precision: \n{precision:.4f}")
            print(f"\n Weighted Recall: \n{recall:.4f}")
            print(f"\n Weighted F1: \n{f1:.4f}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "confusion_matrix": cm,
                "classification_report": self.report,
                "Y": all_targets,
                "Y_pred": all_preds,
                "Y_probability": all_probabilities,
                "f1_score": f1,
            }

#     Validation eval


        val_results = None

        if val_dl is not None:
            print(f"Evaluating on a validation set...")

            val_results = evaluate_dataset(classifier, val_dl)

            print(f"\n Validation Accuracy: "
                  f"{val_results['accuracy']: 4f} "
                  f"({val_results['accuracy'] * 100:.2f}%))")

            print(f"\n Validation Classification Report:\n"
                  f"\n{val_results['classification_report']}")

            print(f"\n Validation Confusion Matrix:\n"
                  f"\n{val_results['confusion_matrix']}")

            print(f"\n Validation Weighted Precision: "
                  f"\n{val_results['precision']:.4f}")

            print(f"\n Validation Weighted Recall: "
                  f"\n{val_results['recall']:.4f}")

            print(f"\n Validation Weighted F1: "
                  f"\n{val_results['f1']:.4f}")

#             Test eval

        test_results = None

        if test_dl is not None:
            print(f"Evaluating on a test set...")

            test_results = evaluate_dataset(classifier, test_dl)

            print(f"\n Test Accuracy: "
                  f"{test_results['accuracy']: 4f} "
                  f"({test_results['accuracy'] * 100:.2f}%))")

            print(f"\n Test Classification Report:\n"
                  f"\n{test_results['classification_report']}")

            print(f"\n Test Confusion Matrix:\n"
                  f"\n{test_results['confusion_matrix']}")

            print(f"\n Test Weighted Precision: "
                  f"\n{test_results['precision']:.4f}")

            print(f"\n Test Weighted Recall: "
                  f"\n{test_results['recall']:.4f}")

            print(f"\n Test Weighted F1: "
                  f"\n{test_results['f1']:.4f}")

#             Store model

        self.activeModel = classifier

        if test_results is not None:
            self._last_accuracy = test_results['accuracy']
            self.report = test_results['classification_report']
        elif val_results is not None:
            self._last_accuracy = val_results['accuracy']
            self.report = val_results['classification_report']
        else:
            self._last_accuracy = 0.0
            self.report = None
        self.status = "ready"

        return {
            "model": self.activeModel,
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
            "train_accuracy": (
                loss_history["accuracy"]
                if loss_history is not None
                else None
            ),

            "val_accuracy": (
                val_results["accuracy"]
                if val_results is not None
                else None
            ),

            "test_accuracy": (
                test_results["accuracy"]
                if test_results is not None
                else None
            ),

            "validation_precision": (
                val_results["precision"]
                if val_results is not None
                else None
            ),

            "validation_recall": (
                val_results["recall"]
                if val_results is not None
                else None
            ),

            "validation_f1": (
                val_results["f1"]
                if val_results is not None
                else None
            ),

            "validation_confusion_matrix": (
                val_results["confusion_matrix"]
                if val_results is not None
                else None
            ),

            "validation_classification_report": (
                val_results["classification_report"]
                if val_results is not None
                else None
            ),

            "test_precision": (
                test_results["precision"]
                if test_results is not None
                else None
            ),

            "test_recall": (
                test_results["recall"]
                if test_results is not None
                else None
            ),

            "test_classification_report": (
                test_results["classification_report"]
                if test_results is not None
                else None
            ),

            "test_confusion_matrix": (
                test_results["confusion_matrix"]
                if test_results is not None
                else None
            ),

            "test_f1": (
                test_results["1"]
                if test_results is not None
                else None
            ),

            "loss_history": loss_history,
            "X_test": X_test,
            "X_val": X_val,
            "Y_val": Y_val,
            "Y_test": Y_test,
            "Y_pred": (
                test_results["Y_pred"]
                if test_results is not None
                else None
            ),

            "Y_probability": (
                test_results["Y_probability"]
                if test_results is not None
                else None
            ),
            "Y_val_probability": (
                val_results["Y_probability"]
                if val_results is not None
                else None
            ),

            "Y_val_pred": (
                val_results["Y_pred"]
                if val_results is not None
                else None
            ),

        }





#       CLASSIFY

    def classify(self, image_path: str, threshold: float = CONFIDENCE_THRESHOLD) -> ClassificationResult:
        if self.activeModel is None:
            raise RuntimeError("No model loaded")

        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        residuals = self.srm.extract_noise_residuals(img, normalise=False)  # apply filters

        # crop the centre
        h, w = img.shape[:2]

        if h < self.img_size or w < self.img_size:
            raise ValueError(
                f"Image must be at least: "
                f"{self.img_size} x {self.img_size}."
                f"Received: {h}x{w}."
            )

        # crop image
        start_h = (h - self.img_size) // 2
        start_w = (w - self.img_size) // 2

        resized_channels = []
        # crop to fixed spatial size 4 ViT path embedding
        for res in residuals:
            crop = res[start_h:start_h + self.img_size, start_w:start_w + self.img_size]
            resized_channels.append(crop)

        # stack res maps - 12x64x64
        stacked = np.stack(resized_channels, axis=0).astype(np.float32)  # combine the res maps into 1 array
        stacked = np.clip(stacked, -3.0, 3.0) / 3.0  # scale to [-1,1] as used during training

        features = torch.from_numpy(stacked).unsqueeze(0).float()  # add batch dimensions - 1x12x64x64
        features = features.to(self.device)

        # Classification

        self.activeModel.eval()

        with torch.no_grad():
            logits = self.activeModel(features)  # run image through model
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # convert logits into probabilities

        # classify
        predicted_idx = int(np.argmax(probs))  # get idx of class w/ highest prob
        confidence = float(probs[predicted_idx])  # use idx to get the prob -> confidence score
        is_flagged = confidence < threshold  # flagged if confidence is below threshold

        return ClassificationResult(
            predictedClass=CLASSES[predicted_idx],
            confidenceScore=confidence,
            isFlagged=is_flagged,
            # map class indices to names for allProbabilities dict
            allProbabilities={
                CLASSES[i]: round(float(p), 4)
                for i, p in enumerate(probs)
                if i in CLASSES
            },
            threshold=threshold, )

    def saveModel(self, file_path: str = "models/cnn_model.pth"):

        if self.activeModel is None:
            raise RuntimeError(f"No active model to save")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        bundle = {
            "model_state_dict": self.activeModel.cpu().state_dict(),
            "img_size": self.img_size,
            "n_classes": self.n_classes,
            "num_channels": self.num_channels,
            "report": self.report,
            "accuracy": self._last_accuracy,
        }

        torch.save(bundle, file_path)
        self.activeModel.to(self.device)
        print(f"CNN Model saved: {file_path}")

    def loadModel(self, file_path: str = "models/cnn_model.pth"):
        # load saved model from disk
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find model at: {file_path}")  # raise if file doesn't exist

        # Load model

        bundle = torch.load(file_path,
                            map_location=self.device,
                            weights_only=False)

        classifier = ManualCNN().to(self.device) # create archietecture
        classifier.load_state_dict(bundle["model_state_dict"]) # load weights
        classifier.eval()

        # Restore pipeline
        self.activeModel = classifier
        self.report = bundle.get("report", None)
        self._last_accuracy = bundle.get("accuracy", 0.0)
        self.img_size = bundle.get("img_size", 64)
        self.n_classes = bundle.get("n_classes", 4)
        self.num_channels = bundle.get("num_channels", 12)
        self.status = "ready"
        print(f" CNN Model loaded from {file_path}")




