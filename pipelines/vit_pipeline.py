import numpy as np # for arrays
import cv2 # image loading
import os, sys
import pickle
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pipelines.pipeline import Pipeline
from transformers import ViTModel
from results.classification_result import ClassificationResult

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

CLASSES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}
CONFIDENCE_THRESHOLD = 0.60

class PretrainedViT(nn.Module):
    # adapts pre-trained ViT for SRM res maps
    def __init__(self, n_classes=4):
        super().__init__()

        # load pretrained model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)

    #     custom classifier head for 4 stego classes
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.RELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # pass 224x224 to ViT
        outputs = self.vit(pixel_values = x)
        # extract the cls token
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def get_embeddings(self,x):
        outputs = self.vit(pixel_vales=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token



class ViT_Pipeline(Pipeline):

    def __init__(self, in_channels=3, n_classes=4):
        super().__init__(name="ViT Pipeline")


        self.report = None
        self.in_channels = in_channels
        self.img_size = 224
        self.patch_size = 16
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_accuracy =0.0

    #     instantiate the model
        self.model = PretrainedViT( n_classes=self.n_classes)
        self.model = self.model.to(self.device) # move model to selected device
        self.activeModel = self.model

    def train(
            self,
            train_dir,
            val_dir=None, test_dir=None,
            epochs=50,
            learning_rate=2e-5,
            batch_size=32,
            max_per_class=None,
            test_size=0.2,
            random_state=42
    ):

        print("\n-------ViT Pipeline: Training...-------")
        print(f"Target channels: {self.in_channels}")
        
        def _load_vit_images(self, data_dir, img_size=224, max_per_class=None):
            
            X,Y = [],[]
            
            for label, directory in data_dir.items():
                
                files = [
                    f for f in os.listdir(directory)
                    if f.lower().endswith(('.pgm', '.png', '.tiff', '.tif'))
                    ]
                
                if max_per_class is not None:
                    files = files[:max_per_class]

                for fname in files:
                    img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)  # load images as grayscale
                    # image validation, skip if unreadable
                    if img is None:
                        print("Couldn't load image:", fname)
                        continue
                    # resize
                    
                    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    
                    img = img.astype(np.float32) / 255.0
                    
                    img = np.stack([img, img, img], axis=0)
                    
                    X.append(img)
                    Y.append(label)
                    
            return (
                np.array(X, dtype=np.float32),
                np.array(Y, dtype=np.int64)
            )

        # Load dataset
        def get_dataset(data_dir, cache_path):

            if data_dir is None:
                return  None, None

            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_path}")
                data = np.load(cache_path)
                return  data["X"], data["Y"]
            

            X, Y = _load_vit_images(data_dir,img_size = self.img_size, max_per_class=max_per_class)
            np.savez(cache_path, X=X, Y=Y)
            print(f"Cached features to {cache_path}")
            return X,Y

        # load training dataset

        print(f"\nLoading training dataset...")

        X_train, Y_train = get_dataset(train_dir, cache_path="vit_cache_train.npz")

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

        if X_train.shape[1] != 3:
            self.status = "idle"
            raise ValueError(
                f"Expected 3 images,"
                f"but received {X_train.shape[1]}"
            )

            # Load validation dataset

        X_val = Y_val = None

        if val_dir is not None:
            print("\n Loading validation dataset...")

            X_val, Y_val = get_dataset(val_dir, cache_path="vit_cache_val.npz")

            print(f"Validation dataset shape{X_val.shape}  ")
            print(f"Validation labels shape{Y_val.shape} ")

        #      Load test data

        X_test = Y_test = None

        if test_dir is not None:
            print("\n Loading test dataset...")

            X_test, Y_test = get_dataset(test_dir, cache_path="vit_cache_test.npz")

            print(f"\nTest dataset shape{X_test.shape}  ")
            print(f"Test labels shape{Y_test.shape} ")


        print(f"X range: [{X_train.min():.3f},{X_train.max():.3f} ]")
        print(f" Classes: {dict(zip(*np.unique(Y_train, return_counts=True)))}")


#        # convert arrays to pytorch tensors
        X_train_t = torch.FloatTensor(X_train)
        Y_train_t = torch.LongTensor(Y_train)

#         create dataloaders
        train_ds = TensorDataset(X_train_t, Y_train_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Validation Dataloader

        val_dl = None

        if X_val is not None:
            X_val_t = torch.from_numpy(X_val).float()
            Y_val_t = torch.from_numpy(Y_val).long()

            val_ds = TensorDataset(X_val_t, Y_val_t)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Test Dataloader

        test_dl = None

        if X_test is not None:
            X_test_t = torch.from_numpy(X_test).float()
            Y_test_t = torch.from_numpy(Y_test).long()

            test_ds = TensorDataset(X_test_t, Y_test_t)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#             Create ViT
        print("\nCreating pretrained ViT")

        self.model = PretrainedViT(n_classes=self.n_classes).to(self.device)
        self.activeModel = self.model

#          Optimization

        optimizer = torch.optim.AdamW([
            {'params': self.model.vit.parameters(), 'lr': 1e-6},
            {'params': self.model.classifier.parameters(), 'lr': 1e-3},
            ], weight_decay=1e-3)

        # Loss

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

#         Training loop

        print(f"\nTraining for {epochs} epochs...")
        loss_history = []
        for epoch in range(epochs):

            self.model.train()

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # loop through each batch
            for batch_X, batch_Y in train_dl:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device) # move the batches to the selected device

#                  clear prev batch's grads
                optimizer.zero_grad()
#                 forward pass
                logits = self.model(batch_X)
#                 compute loss
                loss = criterion(logits, batch_Y)
#                 backpropagation - grad calculation
                loss.backward()
#                 grad clipping to prevent exploding grads
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                update weights
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == batch_Y).sum().item()
                total_samples += batch_X.size(0)

            avg_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples

            # Validation during trainin

            val_epoch_accuracy = None

            if val_dl is not None:
                self.model.eval()

                val_correct = 0
                val_total = 0

                with torch.no_grad():  # disable grad calcs
                    # loop through each batch
                    for batch_X, batch_Y in val_dl:
                        batch_X = batch_X.to(self.device)  # move batch to selected device
                        logits = self.model(batch_X)  # run forward pass - run the model
                        preds = torch.argmax(logits,
                                             dim=-1).cpu().numpy()  # find class with largest logit, move it back to cpu and convert to numpy array
                        val_correct += (preds == batch_Y).sum().item()
                        val_total += batch_Y.size(0)

                val_epoch_accuracy = (val_correct / val_total)

            loss_history.append({
                "loss": avg_loss,
                "accuracy": train_accuracy,
                "val_accuracy": val_epoch_accuracy
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                if val_epoch_accuracy is not None:
                    print(f" Epoch [{epoch + 1}/{epochs}] "
                          f" Loss: {avg_loss:.4f} | "
                          f" Accuracy: {train_accuracy:.4f}"
                          f" Val Accuracy: {val_epoch_accuracy:.4f}"
                          )
                else:
                    print(f" Epoch [{epoch + 1}/{epochs}] "
                          f" Loss: {avg_loss:.4f} | "
                          f" Accuracy: {train_accuracy:.4f}"
                          )

        def evaluate_dataset(self, model, dataloader):

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
                    logits = model(batch_X)  # run forward pass - run the model
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = torch.argmax(logits,
                                         dim=-1).cpu().numpy()  # find class with largest logit, move it back to cpu and convert to numpy array
                    all_preds.extend(preds.tolist())  # add to all predictions
                    all_targets.extend(batch_Y.numpy().tolist())  # add to all true labels
                    all_probabilities.extend(probs.tolist())

            all_targets = np.array(all_targets)
            all_probabilities = np.array(all_probabilities)
            all_preds = np.array(all_preds)
            accuracy = accuracy_score(all_targets, all_preds)
            cm = confusion_matrix(all_targets, all_preds, labels=list(CLASSES.keys()))
            report = classification_report(all_targets, all_preds, labels=list(CLASSES.keys()),
                                           target_names=[CLASSES[i] for i in CLASSES.keys()],
                                           zero_division=0, output_dict=True
                                           )

            return {
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report,
                "Y": all_targets,
                "Y_pred": all_preds,
                "Y_probability": all_probabilities,
            }

            #     Validation eval

        val_results = None

        if val_dl is not None:
            print(f"Evaluating on a validation set...")

            val_results = evaluate_dataset(self.model, val_dl)

            print(f"\n Validation Accuracy: "
                  f"{val_results['accuracy']: 4f} "
                  f"({val_results['accuracy'] * 100:.2f}%))")

            print(f"\n Validation Classification Report:\n"
                  f"\n{val_results['classification_report']}")

            print(f"\n Validation Confusion Matrix:\n"
                  f"\n{val_results['confusion_matrix']}")

        #             Test eval

        test_results = None

        if test_dl is not None:
            print(f"Evaluating on a test set...")

            test_results = evaluate_dataset(self.model, test_dl)

            print(f"\n Test Accuracy: "
                  f"{test_results['accuracy']: 4f} "
                  f"({test_results['accuracy'] * 100:.2f}%))")

            print(f"\n Test Classification Report:\n"
                  f"\n{test_results['classification_report']}")

            print(f"\n Test Confusion Matrix:\n"
                  f"\n{test_results['confusion_matrix']}")


        #             Store model

        self.activeModel = self.model

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
            "model": self.model,
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
                loss_history[-1]["accuracy"]
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


        # classify image using trained model


    def generate_tsne(self, dataloader,max_samples_per_class=500):

        if self.activeModel is None:
            raise RuntimeError("No model available.")

        self.activeModel.eval()

        all_embeddings = []
        all_labels = []

        with torch.no_grad():

            for batch_X, batch_Y in dataloader:
                batch_X = batch_X.to(self.device)
                embeddings = self.activeModel.get_embeddings(batch_X)

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(batch_Y.cpu().numpy())

            embeddings = np.concatenate(all_embeddings, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            selected_indices = []

            for class_id in CLASSES:
                class_indices = np.where(labels == class_id)[0]

                if len(class_indices)> max_samples_per_class:
                    class_indices = class_indices[:max_samples_per_class]
                selected_indices.extend(class_indices)
            selected_indices = np.array(selected_indices)
            embeddings = embeddings[selected_indices]
            labels = labels[selected_indices]

            print(f"T-SNE samples: {len(labels)}")
            print(f"Embeddings shape: {embeddings.shape}")

    #          768 to 50 dims

            n_components = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=42)

            embeddings_pca = pca.fit_transform(embeddings)

    #         50 to 2 dims

            tsne = TSNE(n_components=n_components,perplexity=30 ,random_state=42, init='pca',learning_rate='auto')

            embeddings_tsne = tsne.fit_transform(embeddings_pca)

            print(f"TSNE shape: {embeddings_tsne.shape}")

    #         Plot

            plt.figure(figsize=(10,8))

            for class_id,class_name in CLASSES.items():

                mask = labels == class_id

                plt.scatter(
                    embeddings_tsne[mask, 0],
                    embeddings_tsne[mask, 1],
                    label=class_name,
                    alpha=0.7,
                    s=20
                )

            plt.title("t-SNE Visualisation for ViT Embeddings")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend()
            plt.grid(alpha=0.2)

            plt.tight_layout

            return plt.gcf()




    def classify(self, image_path: str, threshold: float = CONFIDENCE_THRESHOLD) -> ClassificationResult:
        if self.activeModel is None:
            raise RuntimeError("No model loaded")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0
        img = np.stack((img,)*3, axis=0)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0).to(self.device)

        self.activeModel.eval()
        with torch.no_grad():
            logits = self.activeModel(tensor) # run image through model
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0] # convert logits into probabilities

        # classify
        predicted_idx = int(np.argmax(probs)) # get idx of class w/ highest prob
        confidence = float(probs[predicted_idx]) # use idx to get the prob -> confidence score
        is_flagged = confidence < threshold # flagged if confidence is below threshold


        return ClassificationResult(
            predictedClass=CLASSES[predicted_idx],
            confidenceScore=confidence,
            isFlagged = is_flagged,
            # map class indices to names for allProbabilities dict
            allProbabilities={
                CLASSES[i]: round(float(p), 4)
                for i, p in enumerate(probs)
                if i in CLASSES
            },
            threshold=threshold,)

    def saveModel(self, file_path: str = "models/vit_model.pkl"):

        if self.activeModel is None:
            raise RuntimeError(f"No active model to save")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        bundle = {
            "state_dict": self.activeModel.cpu().state_dict(),
            "img_size": 224,
            "n_classes": self.n_classes,
            "in_channels": 3,
            "patch_size":  16,
            "report": self.report,
            "accuracy": self._last_accuracy,
        }

        with open(file_path, "wb") as f:
            pickle.dump(bundle, f)
        self.activeModel.to(self.device)
        print(f" Model saved: {file_path}")

    def loadModel(self, file_path: str = "models/vit_model.pkl"):
        # load saved model from disk
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find model at: {file_path}")  # raise if file doesn't exist

        with open(file_path, "rb") as f:
            bundle = pickle.load(f)

        self.img_size = bundle.get("img_size",224)
        self.patch_size = bundle.get("patch_size",16)
        self.in_channels = bundle.get("in_channels",3)
        self.n_classes = bundle.get("n_classes",4)

        # reinstantiate the pretrained model
        model = PretrainedViT( n_classes=self.n_classes).to(self.device)
        model.load_state_dict(bundle["state_dict"])
        model.eval()

        self.activeModel = model
        self.report = bundle.get("report", None)
        self._last_accuracy = bundle.get("accuracy", 0.0)
        self.status = "ready"
        print(f" Model loaded from {file_path}")







