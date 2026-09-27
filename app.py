# Imports

import streamlit as st  # UI
import numpy as np # for arrays
import cv2 # image loading
import matplotlib.pyplot as plt # for plotting
import matplotlib
import tkinter as tk
import pandas as pd

from pipelines.pipeline import modelExists, Pipeline
from tkinter import filedialog

matplotlib.use('Agg') # server-side rendering
import sys, os, io # sys path,file ops and serialization

from pipelines.rf_pipeline import RF_Pipeline
from pipelines.vit_pipeline import ViT_Pipeline
from pipelines.mlp_pipeline import MLP_Pipeline
from pipelines.cnn_pipeline import CNN_Pipeline

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title='Steganography Analysis',
                   page_icon="🫆",
                   layout="wide",
                   initial_sidebar_state="expanded"


)


# CSS
st.markdown("""
<style>
   
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 8px;
    }
    .result-box{
        background-color: #1a1a2e;
        color: #000;
        padding: 80px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight:bold;
        margin: 0px
    }
    
    .flag-box {
        background-color: #c0392b;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        
        }
        
    .confidence-box {
        background-color: #2c3e50;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
 </style>
""", unsafe_allow_html=True)

# Constants
RF_MODEL_PATH = "models/rf_model.pkl"
MLP_MODEL_PATH = "models/hpf_mlp_model.pkl"
ViT_MODEL_PATH = "models/vit_model.pkl"
CNN_MODEL_PATH = "models/cnn_model.pth"
# training data directories

TRAIN_DIR = {
    0: "data/splits/train/cover",
    1: "data/splits/train/stego_lsb",
    2: "data/splits/train/stego_wow",
    3: "data/splits/train/stego_suniward",
}

VAL_DIR = {
    0: "data/splits/val/cover",
    1: "data/splits/val/stego_lsb",
    2: "data/splits/val/stego_wow",
    3: "data/splits/val/stego_suniward",
}

TEST_DIR = {
    0: "data/splits/test/cover",
    1: "data/splits/test/stego_lsb",
    2: "data/splits/test/stego_wow",
    3: "data/splits/test/stego_suniward",
}

CLASS_NAMES = {0: "Cover", 1: "LSB", 2: "WOW", 3: "S-UNIWARD"}

os.makedirs("models", exist_ok=True)  # validation

# Header

st.title("Steganography Analysis")
st.markdown("Detection and Classification of Steganographic Embedding Algorithms")
st.markdown("222117186 | 260IT28X30 | IT7 Project ")
st.divider()

# Tkinter Folder picker dialog func
def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_selected = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_selected

# sidebar

with (st.sidebar):
    st.header("Configuration")

# SECTION 1
    st.subheader("1. Pipeline")
    pipeline_choice = st.selectbox(
        "Detection Strategy",
        ["Random Forest", "HPF + MLP", "Vision Transformer (ViT)", "Convolutional Neural Network (CNN)", "Run All Pipelines"]
    )

    # match model path to selected pipeline
    is_rf =  pipeline_choice == "Random Forest"
    is_mlp =  pipeline_choice == "HPF + MLP"
    is_vit =  pipeline_choice == "Vision Transformer (ViT)"
    is_cnn = pipeline_choice == "Convolutional Neural Network (CNN)"
    is_all = pipeline_choice == "Run All Pipelines"

    if is_rf:
        active_model_path = RF_MODEL_PATH
    elif is_mlp:
        active_model_path = MLP_MODEL_PATH
    elif is_vit:
        active_model_path = ViT_MODEL_PATH
    elif is_cnn:
        active_model_path = CNN_MODEL_PATH
    else:
        active_model_path = None

    st.divider()

# SECTION 2
    st.header("2. Model")

    if is_all:
        st.info("Run All Pipelines uses the saved model for each pipeline.")

        all_model_paths = {
        "Random Forest": RF_MODEL_PATH,
        "HPF + MLP": MLP_MODEL_PATH,
        "Vision Transformer (ViT)": ViT_MODEL_PATH,
        "Convolutional Neural Network (CNN)":CNN_MODEL_PATH,
        }

        for name, path in all_model_paths.items():
            if os.path.exists(path):
                st.success(f"{name}: saved model found")
            else:
                st.warning(f"{name}: model not found")

        model_action = "Load saved models"
        train_button = False

    else:

        model_exists = os.path.exists(active_model_path)

            # validate
        if model_exists:
            st.success("Saved model exists")
            model_action = st.radio(
                "Model Action",
                ["Load saved model", "Retrain model"],
                index =0
            )
        else:
            st.warning("No saved model exists")
            model_action = "Retrain model"


    # only for training
        if model_action == "Retrain model":
            if is_rf:
                n_trees = st.slider("Number of Trees", 50, 500, 100, step=50)
            elif is_mlp:
                epochs = st.slider("Number of Epochs (MLP)", 10, 200, 50, step=10)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001,0.005,0.01],
                    value=0.01,
                )
            elif is_vit:
                epochs = st.slider("Number of Epochs ", 10, 200, 50, step=10)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.00002, 0.0001, 0.0005, 0.001,0.005,0.01],
                    value=2e-5,
                )

            elif is_cnn:
                epochs = st.slider("Number of Epochs (CNN) ", 10, 100, 30, step=10)
                learning_rate = st.select_slider(
                    "Learning Rate (CNN)",
                    options=[0.00001, 0.0005, 0.001, 0.005],
                    value=0.001,
                )

            train_button = st.button("Train & Save Model", use_container_width=True, type="primary")
        else:
            train_button = False

    st.divider()


# SECTION 3

    st.sidebar.subheader("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select mode",
        ["Single Image", "Local Folder"],
        help="Single: classify one image. Local Folder: allows direct parsing of local evidence folders."
    )

    folder_path = ""
    expected_class = "Unknown / Mixed"

    if analysis_mode == "Single Image":
        st.subheader("3. Upload Image")
        uploaded_file = st.file_uploader(
            "Select image for analysis",
            type=["pgm", "png", "tiff", "jpeg", "jpg"]
        )
    else:
        st.subheader("3. Local Folder")
        if st.button("Browse Folder...", use_container_width=True):
            folder_selected = select_folder()
            if folder_selected:
                st.session_state["folder_path"] = folder_selected

        folder_path = st.session_state.get("folder_path", "")
        st.text_input(
            "Selected Evidence Folder Path",
            value=folder_path,
            disabled=True,
            placeholder="No folder selected",
            help="Absolute path to the directory with the evidence images"
        )
        expected_class = st.selectbox(
            "Expected Baseline Class",
            ["Unknown / Mixed", "Cover", "LSB", "WOW", "S-UNIWARD"],
            help="If the folder should contain only clean images, any stego prediction will be flagged as an anomaly."

        )
        uploaded_file = None

    st.divider()


# SECTION 4

    st.subheader("4. Confidence Threshold")
    threshold = st.slider(
        "Flag below this confidence",
        min_value=0.25,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help= "Results below this threshold are flagged for manual review"

    )

# SECTION 5

    st.subheader("5. Run")

    if analysis_mode == "Single Image":
        run_disabled = uploaded_file is None
    else:
        run_disabled = not (folder_path and os.path.exists(folder_path))

    run_button = st.button(
        "Run Analysis",
        use_container_width=True,
        type="primary",
        disabled=run_disabled,
    )
    st.divider()


# pipelines

@st.cache_resource # ensures below section runs once per session i.e single pipeline instantiation
def get_rf_pipeline() -> RF_Pipeline:
    # creates and returns a cached RF pipeline instance
    return RF_Pipeline()


@st.cache_resource
def get_mlp_pipeline() -> MLP_Pipeline:
    # creates and returns a cached MLP pipeline instance
    return MLP_Pipeline()

@st.cache_resource
def get_vit_pipeline() -> ViT_Pipeline:
    # creates and returns a cached ViT pipeline instance
    return ViT_Pipeline()

@st.cache_resource
def get_cnn_pipeline() -> CNN_Pipeline:
    # creates and returns a cached CNN pipeline instance
    return CNN_Pipeline()

def getActivePipeline():
    if is_rf:
        return get_rf_pipeline()
    elif is_mlp:
        return get_mlp_pipeline()
    elif is_vit:
        return get_vit_pipeline()
    elif is_cnn:
        return get_cnn_pipeline()
    else:
        return None

# get all pipelines helper method
def getAllPipelines():
    return {
        "Random Forest":{
            "pipeline": get_rf_pipeline(),
            "model_path": RF_MODEL_PATH,
        },
        "HPF + MLP":{
            "pipeline": get_mlp_pipeline(),
            "model_path": MLP_MODEL_PATH,
        },
        "Vision Transformer (ViT)":{
            "pipeline": get_vit_pipeline(),
            "model_path": ViT_MODEL_PATH
        },
        "Convolutional Neural Network (CNN)":{
            "pipeline": get_cnn_pipeline(),
            "model_path": CNN_MODEL_PATH
        }
    }

# all pipelines classification helper
def classifyWithAllPipelines(image_path, threshold):
    all_results = {}
    pipelines = getAllPipelines() # get all pipelines

    for name, info in pipelines.items():

        pipeline = info["pipeline"]
        model_path = info["model_path"]

        try:
            if not pipeline.isReady():
                if modelExists(model_path):
                    pipeline.loadModel(model_path)
                else:
                    all_results[name] = {
                        "error": f"Model not found: {model_path}",
                    }
                    continue

            result = pipeline.classify(image_path, threshold=threshold)

            all_results[name] = result

        except Exception as e:
            all_results[name] = {"error": str(e)}

    return all_results

def displayAllPipelineResults(all_results):

    st.subheader("Pipeline Comparison")

    summary_rows = []

    for pipeline_name, result in all_results.items():
        # Validation
        if result is None:
            continue

        if isinstance(result, dict) and "error" in result:
            summary_rows.append({
                "Pipeline": pipeline_name,
                "Prediction": "unavailable",
                "Confidence": "-",
                "Status": result["error"],
            })
            continue

        confidence = float(result.confidenceScore)

        summary_rows.append({
            "Pipeline": pipeline_name,
            "Prediction": result.predictedClass,
            "Confidence": f"{confidence * 100:.2f}%",
            "Status": "Flagged" if result.isFlagged else "Normal",
        })

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Class probabilities")
    classes_ordered = ["Cover", "LSB", "WOW", "S-UNIWARD"]

    for pipeline_name, result in all_results.items():
        # Validation
        if result is None:
            continue

        if isinstance(result, dict) and "error" in result:
            continue

        st.markdown(f"{pipeline_name}")
        probabilities = result.allProbabilities

        probability_rows = []

        for class_name in classes_ordered:
            probability_rows.append({
                "Class": class_name,
                "Probability": f"{float(probabilities.get(class_name, 0)) * 100:.2f}%",
            })

        st.dataframe(
            pd.DataFrame(probability_rows),
            use_container_width=True,
            hide_index=True
        )



def generate_embedding_heatmap(residuals, img):
    res_stack = np.array(residuals) # convert into an array
    res_mean = np.mean(res_stack, axis=0)

    col_1, col_2 = st.columns(2)

    # OG image
    with col_1:
        st.subheader("Original Image")
        st.image(img,
                 caption=f"{uploaded_file.name}",
                 use_container_width=True,
                 clamp=True)
        # Mean Residual
    with col_2:
        st.subheader("Mean Residual")
        fig2, ax2 = plt.subplots(figsize=(16, 4))
        fig2.patch.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")
        ax2.imshow(res_mean, cmap='hot', vmin=0, vmax=255)
        ax2.axis('off')
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# For training

if train_button:

    if "coming soon" in pipeline_choice:
        st.error(f"Pipeline coming soon. Not yet implemented.")
        st.stop()

    st.header("Model Training")

    # validation
    missing = [d for d in TRAIN_DIR.values() if not os.path.exists(d) or len(os.listdir(d)) == 0]

    if missing:
        st.error(f"Missing or empty data directories.\n {missing} ")
        st.stop()

    progress_bar = st.progress(0, text="Training commencing...")

    with st.spinner(f"Training {pipeline_choice} pipeline..."):
        progress_bar = st.progress(10, text="Loading and preprocessing images...")

        pipeline = getActivePipeline()


        # call train() with pipeline params
        if is_rf:
            results = pipeline.train(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir = TEST_DIR, n_estimators=n_trees)
        elif is_mlp:
            results = pipeline.train(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir = TEST_DIR, epochs=epochs, learning_rate=learning_rate)
        elif is_vit:
            results = pipeline.train(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir = TEST_DIR, epochs=epochs, learning_rate=learning_rate)
        elif is_cnn:
            results = pipeline.train(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir = TEST_DIR, epochs=epochs, learning_rate=learning_rate)

        progress_bar = st.progress(80, text="Saving model...")

        pipeline.saveModel(file_path=active_model_path)

        # call saveModel on pipeline

        # clear cache for next load
        get_rf_pipeline.clear()
        get_mlp_pipeline.clear()
        get_vit_pipeline.clear()
        get_cnn_pipeline.clear()

        progress_bar = st.progress(100, text="Training complete!")

    st.success(f"Model trained and saved to {active_model_path}")

    # Metrics displaying

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{results['accuracy']:.1%}")
    col2.metric("Pipeline", pipeline_choice.split()[0])
    col3.metric("Classes", "4")

    # loss curve for mlp

    if (is_mlp ) and "loss_history" in results:
        st.subheader("Training Loss Curve")
        fig_loss, ax_loss = plt.subplots(figsize=(6, 3))
        ax_loss.plot(results["loss_history"], color="#e74c3c")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        plt.tight_layout()
        st.pyplot(fig_loss)
        plt.close()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]

    # show class names present in test set
    present_classes = sorted(set(results["Y_test"].tolist()
                                 if hasattr(results["Y_test"], "tolist")
                                 else results["Y_test"]))
    class_names = [CLASS_NAMES[i] for i in present_classes]

    fig_cm,ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels = class_names,
        yticklabels = class_names,
        xlabel = "Predicted",
        ylabel = "True",
        title = f" {pipeline_choice} - Confusion Matrix",
    )
    plt.setp(ax_cm.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j,i,str(cm[i,j]), ha="center", va="center", color="white"
                       if cm[i,j] > thresh else "black")

    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

    st.divider()


# main area
if analysis_mode == "Single Image":

# Single Image Mode
    if uploaded_file is None:
        st.info("Upload an image and use the sidebar to configure a pipeline.")

        col1, col2, col3, = st.columns(3)
        with col1:
            st.markdown("""
            **Supported Formats**
            - PGM
            - PNG
            - Grayscale or color images
            """)

        with col2:
            st.markdown("""
            **Detection Classes**
            - Cover
            - LSB
            - WOW
            - S-UNIWARD
            """)

        with col3:
            st.markdown("""
            **Pipelines**
            - Random Forest
            - High Pass Filters (HPF) + MultiLayer Perceptron (MLP)
            - ViT
            - CNN 
            """)
    else:
        # load images
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            st.error("Image cannot be loaded. Please double check the format.")
            st.stop()

          #  Preprocessing stage
        st.header("Preprocessing")
        st.markdown(
            "SRM high-pass filters supress image content and amplify steganographic noise residuals."
            "The residual map is sent to the classifier which then performs classification."
        )

        pipeline = getActivePipeline()

        if pipeline is not None:
            residual = pipeline.srm.extract_noise_residuals(img_gray, normalise=False)
        else:
            # instantiate a standalone SRM filter for display
            from preprocessing.srm_filters import  SRMFilter
            residual = SRMFilter().extract_noise_residuals(img_gray, normalise=False)

        col1, col2= st.columns(2)


        with col1:
            st.subheader("Original Image")
            st.image(img_gray,
                     caption=f"{uploaded_file.name} | ",
                     use_container_width=True,
                     clamp=True)
        with col2:
            st.subheader("SRM Residual Map")
            fig, ax =plt.subplots(figsize=(16, 4))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            ax.imshow(residual[0], cmap='inferno')
            ax.set_title("After SRM filters", color = "white", fontsize = 10)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(
                     # f"Mean: {residual[0].mean():.2f} | "
                     f"Std: {residual[0].std():.2f} | "
                     f"Range: [{residual[0].min():.1f} , {residual[0].max():.1f}] "
                     )

        st.divider()

    # Classification

        st.header("Classification")

        if run_button:

            # run all pipelines

            if is_all:
                temp_path = f"temp_{uploaded_file.name}"  # save uploaded image temporarily
                cv2.imwrite(temp_path, img_gray)

                try:
                    with st.spinner("Running image through all 4 pipelines..."):
                        all_results = classifyWithAllPipelines(temp_path,threshold)

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                # display comparison
                displayAllPipelineResults(all_results)
                st.divider()

                valid_results = [
                    result
                    for result in all_results.values()
                    if not isinstance(result, dict) and result is not None
                ]

                if valid_results:
                    predictions = [
                        result.predictedClass
                        for result in valid_results
                    ]

                    counts = pd.Series(predictions).value_counts()
                    most_common_class = counts.index[0]
                    agreement = int(counts.iloc[0])

                    st.subheader("Pipeline Agreement")

                    st.caption(
                        f"{len(valid_results)} of {len(all_results)} pipelines "
                        "successfully produced a prediction."
                    )

                    st.metric(
                        "Most commonly predicted class",
                        most_common_class,
                    )

                    st.metric(
                        "Pipeline Agreement",
                        f"{agreement}/{len(valid_results)}",
                    )

            #     Single pipeline

            else:

                pipeline = getActivePipeline()

                # load saved model

                if not pipeline.isReady():
                    if modelExists(active_model_path):
                        pipeline.loadModel(active_model_path)

                    else:
                        st.error(f"No trained model found.")
                        st.stop()
                report = pipeline.report

                with st.spinner(f"Running {pipeline_choice} pipeline..."):
                    # write uploaded image to temp file becoz classify() wants a file path
                    temp_path = f"temp_{uploaded_file.name}"
                    cv2.imwrite(temp_path, img_gray)

                    try:
                        # classify() returns a ClassificationResult object
                        result = pipeline.classify(temp_path, threshold=threshold)
                    finally:
                        # ditch temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                # show result from ClassificationResult object
                predictedClass = result.predictedClass
                confidenceScore = result.confidenceScore
                isFlagged= result.isFlagged
                allProbabilities = result.allProbabilities

                # Display Results

                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    st.subheader("Predicted Class")

                    color = {
                        "Cover": "green",
                        "LSB": "yellow",
                        "WOW": "orange",
                        "S-UNIWARD": "purple",
                    }.get(predictedClass, "#2c3e50")

                    st.markdown(
                        f'<div class="result-box" style="background-color: {color};">'
                        f'{predictedClass}</div>',
                        unsafe_allow_html=True,
                    )

                    if isFlagged:
                        st.markdown(
                            f'<div class="flag-box"> LOW CONFIDENCE - Manual Review Recommended<br>'
                            f'Score {confidenceScore:.1%} is below threshold {threshold:.0%}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.success(f"Confidence: {confidenceScore:.1%} - above threshold {threshold:.0%}")


                with col_res2:
                    st.subheader("Class Probabilities")

                    classes_ordered = ["Cover", "LSB", "WOW", "S-UNIWARD"]
                    colors_ordered = ["green", "yellow", "orange", "purple"]

                    # show only classes the model knows
                    available = [c for c in classes_ordered if c in allProbabilities]
                    probs_value = [allProbabilities.get(c , 0.0)for c in available]
                    bar_colors = [colors_ordered[classes_ordered.index(c)] for c in available]

                    fig3, ax3 = plt.subplots(figsize=(5, 3))
                    bars = ax3.barh(available, probs_value, color=bar_colors, alpha=0.85)

                    # red dash reps current confidence threshold
                    ax3.axvline(x=threshold, color="red", linestyle="dashed", linewidth=1.5, label=f"Threshold ({threshold})")
                    ax3.set_xlabel("Probability", fontsize = 9)
                    ax3.set_xlim(0,1)
                    ax3.legend(fontsize = 8)
                    ax3.tick_params(labelsize=9)

                    # highlight predicted class with gold
                    if predictedClass in available:
                        pred_idx = available.index(predictedClass)
                        bars[pred_idx].set_edgecolor('gold')
                        bars[pred_idx].set_linewidth(2.5)


                    # add prob labels
                    for bar, cls in zip(bars, available):
                        ax3.text(bar.get_width() + 0.01,
                                 bar.get_y() + bar.get_height() / 2,
                                 f'{allProbabilities.get(cls, 0):.3f}',
                                 va='center', fontsize=9,)

                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close()

                st.divider()

                #Metrics summary

                st.header("Evaluation Metrics")
                m1, m2, m3, m4 = st.columns(4)

                m1.markdown("Predicted Class", help ="The class the model assigned the highest probability to.")
                m1.write(predictedClass)
                m2.metric("Confidence Score",   f"{confidenceScore:.1%}",
                          help =("The probability the model assigned to the predicted class" 
                                "Ranges from 0% to 100%. Scores below 60% are flagged for manual review."))
                m3.markdown("Pipeline", help="The detection strategy used to classify this image.")
                m3.write(pipeline_choice)
                m4.metric("Flagged", "Yes" if isFlagged else "No", help="Whether the confidence score fell below the chosen threshold.")
                st.divider()

                st.subheader("Reliability")

                sorted_probs = sorted(allProbabilities.values(), reverse=True)
                margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

                if confidenceScore >= 0.80 and margin >= 0.40:
                    certainty = "High"
                elif confidenceScore >= 0.60 and margin >= 0.20:
                    certainty = "Medium"
                else:
                    certainty = "Low"

                r1, r2 = st.columns(2)
                r1.metric("Decision margin", f"{margin:.1%}", help="Gap between the top and second highest class probability.")
                r2.metric("Certainty", certainty, help="Overall reliability of this result based on confidence and margin.")

                st.divider()

                st.write("")
                st.subheader("Model Eval Metrics")
                col1, col2, col3, col4 = st.columns(4)
                if report is not None and isinstance(report, dict):
                    col1.metric("Accuracy", f"{pipeline._last_accuracy * 100:.2f}%")
                    col2.metric("Precision", f"{report['weighted avg']['precision']*100:.2f}%")
                    col3.metric("Recall", f"{report['weighted avg']['recall']*100:.2f}%")
                    col4.metric("F1-Score", f"{report['weighted avg']['f1-score']*100:.2f}%")

                st.divider()

                # Steg Heatmap Display

                st.subheader("Embedding Location Heatmap")
                st.caption(
                    "Shows where steganographic changes are most concentrated. "
                    "LSB = uniform distribution. WOW/S-UNIWARD = concentrated in textured regions. "
                )
                generate_embedding_heatmap(residual, img_gray)



                # export report

                st.header("Export Report")

                base_report = result.toReport()
                full_report = f""" STEGANALYSIS INVESTIGATION REPORT
                ---------------------------
                Student No. : 222117186
                Module Code : 260IT28X30 IT7 Project
                
                Image       : {uploaded_file.name}
                Pipeline    : {pipeline_choice}
                Threshold   : {threshold:.0%}
                
                {base_report}
                PREPROCESSING
                -----------------
                Image Dimensions  : {img_gray.shape[1]} x {img_gray.shape[0]} px
                Residual mean     : {residual[0].mean():.4f}
                Residual std      : {residual[0].std():.4f}
                Filters applied   : SRM high-pass filters
                Normalisation     : [0, 255]
                
                MODEL
                _____
                Model path        : {active_model_path}
                """

                st.download_button(
                    label="Download Report (.txt)",
                    data=full_report,
                    file_name=f"steganalysis_report_{uploaded_file.name}.txt",
                    mime="text/plain",
                    use_container_width=True,
            )

        else:
            st.info("Click **Run Analysis** in the sidebar to classify image ")
            st.markdown(
                f"**Image loaded:** `{uploaded_file.name}` | "
                f"**Size:** `{img_gray.shape[1]} x {img_gray.shape[0]} px` | "
                f"**Pipeline:** `{pipeline_choice}` | "
            )

elif analysis_mode == "Local Folder":

#     Batch mode
    st.header("Local Directory Classification")

    # Validation
    if not folder_path:
        st.info("Click Browse Folder... in the sidebar to choose a directory on your machine.")
    elif not os.path.exists(folder_path):
        st.error(f"Specified folder path does not exist: {folder_path}")
    elif run_button:

        supported = (
            ".pgm",
            ".png",
            ".tiff",
            ".tif",
            ".jpg",
        )



#         Validation
        if is_all:

            image_paths = []

            for root, _, files in os.walk(folder_path):

                for file in files:

                    if(
                        file.lower().endswith(supported) and not file.startswith(".")
                    ):
                        image_paths.append(os.path.join(root, file))

            if not image_paths:
                st.error(f"No supported image files found in {folder_path}.")
                st.stop()

            st.info(
                f"Found {len(image_paths)} supported image files. "
                "Running all pipelines..."
            )

            progress = st.progress(0, text="Starting batch classification...")

            results = []
            failed = []

            # classification loop

            for i, img_path in enumerate(image_paths):
                filename = os.path.basename(img_path)
                if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                    progress.progress(
                        (i + 1) / len(image_paths),
                        text = (
                            f"Classifying images..."
                            f"({i + 1}/{len(image_paths)})"
                        )
                    )

                try:
                    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img_gray is None:
                        failed.append((filename, "Could not decode image"))
                        continue

                    all_results = classifyWithAllPipelines(img_path, threshold)
                    row = {"filename": filename}

                    predictions = []

                    for pipeline_name, result in all_results.items():

                        if isinstance(result, dict) and "error" in result:
                            row[
                                f"{pipeline_name} Prediction"
                            ] = "Unavailable"

                            row[
                                f"{pipeline_name} Confidence"
                            ] = np.nan

                        else:

                            prediction = result.predictedClass

                            row[
                                f"{pipeline_name} Prediction"
                            ] = result.predictedClass

                            row[
                                f"{pipeline_name} Confidence"
                            ] = result.confidenceScore

                            prediction.append(prediction)

                    if predictions:
                        counts = pd.Series(predictions).value_counts()

                        row["Agreement"] = (
                            f"{counts.iloc[0]/len(predictions)}"
                        )

                    else:

                        row["Agreement"] = "Unavailable"

                    results.append(row)

                except Exception as e:

                    print(f"Error processing image"
                          f"{filename}: {e}")

                    failed.append((filename, str(e)))


            progress.progress(1.0, text="All-pipeline bath classification complete!")

        #         display run-all results

            if not results:
                st.error("No images were successfully classified.")
                st.stop()

            df = pd.DataFrame(results)
            st.divider()

            with st.expander(
                f"View detailed results ({len(df):,} images)"
            ):

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

    #         Failed images

            if failed:
                st.warning(f"{len(failed)} images could not be processed.")

                failed_df = pd.DataFrame(failed, columns=["Filename", "Error"])

                with st.expander("View failed images"):
                    st.dataframe(
                        failed_df,
                        use_container_width=True,
                        hide_index=True,
                    )

    #                 Batch summary

            st.divider()

            st.subheader("Pipeline Prediction Summary")

            pipeline_names = [
                "Random Forest",
                "HPF + MLP",
                "Vision Transformer (ViT)",
                "Convolutional Neural Network (CNN)",
            ]

            for pipeline_name in pipeline_names:
                prediction_column = (f"{pipeline_name} Prediction")

                if prediction_column not in df.columns:
                    continue

                st.markdown(f"{pipeline_name}")

                counts = (
                    df[prediction_column].value_counts()
                )

                summary_df = counts.rename("Number of Images").reset_index()
                summary_df .columns = ["Predicted Class", "Number of Images"]

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )



        else:

            pipeline = getActivePipeline()

        #             Load the model
            if not pipeline.isReady():
                if modelExists(active_model_path):
                    pipeline.loadModel(active_model_path)
                else:
                    st.error(f"No trained model found at {active_model_path}.")
                    st.stop()

#                 Go through directory, checking ext
            image_paths = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(supported) and not file.startswith("."):
                        image_paths.append(os.path.join(root, file))

            if not image_paths:
                st.error(f"No supported image files found in {folder_path}.")
                st.stop()

            st.info(f"Found {len(image_paths)} supported image files. Starting classification...")
            progress = st.progress(0, text="Starting batch classification...", )

            results = []
            failed = []

#                 Classfy each image
            for i, img_path in enumerate(image_paths):
                filename = os.path.basename(img_path)

    #                     update progress
                progress.progress(
                    (i+1)/len(image_paths),
                    text=(
                        f"Classifying {filename}..."
                        f"({i+1}/{len(image_paths)})"))

                try:
    #                         read the images from zip file
                    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    #                         Validation
                    if img_gray is None:
                        failed.append((filename, "Could not decode image"))
                        continue

    #                     Classification
                    result = pipeline.classify(img_path, threshold=threshold)

#                     store results
                    h, w = img_gray.shape

                    # Anomaly determination
                    is_anomaly = False
                    if expected_class != "Unknown / Mixed":
                         is_anomaly = (result.predictedClass != expected_class)

                    results.append(
                        {"filename": filename,
                         "predicted_class": result.predictedClass,
                         "confidence": result.confidenceScore,
                         "flagged": result.isFlagged,
                         "is_anomaly": is_anomaly,
                         "cover_prob": result.allProbabilities.get("Cover", 0),
                         "lsb_prob": result.allProbabilities.get("LSB", 0),
                         "wow_prob": result.allProbabilities.get("WOW", 0),
                         "s-uniward_prob": result.allProbabilities.get("S-UNIWARD", 0),
                         "width": w,
                         "height": h,
                         })

                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                    failed.append((filename, str(e)))

        progress.progress(1.0, text="Batch classification complete!")

#                     Dsiplay results
        if not results:
            st.error("No images were successfully classified!")
            st.stop()

        df = pd.DataFrame(results)

#                 Summary

        st.divider()

        st.subheader("Batch Summary")

        total = len(results)
        flagged = int(df["flagged"].sum())
        class_counts = (df["predicted_class"].value_counts())

        if expected_class != "Unknown / Mixed":
            anomalies = int(df["is_anomaly"].sum())

            s1, s2, s3, s4 = st.columns(4)

            s1.metric("Total Classified",total)
            s2.metric("Expected Baseline",expected_class)
            s3.metric("Anomalies Detected",anomalies, delta=f"{anomalies} non-{expected_class}", delta_color="inverse")
            s4.metric("Low Confidence Flags",flagged)

            if anomalies > 0:
                st.warning(f"Detected {anomalies} image(s) deviating from the expected baseline {expected_class}.")

        else:
            s1, s2, s3 = st.columns(3)

            s1.metric("Total Classified", total)
            s2.metric("Low Confidence Flags", flagged)
            s3.metric("Failed", len(failed))

#                     Class Distribution
        st.divider()
        st.subheader("Class Distribution")

        fig_dist, ax_dist = plt.subplots(figsize=(7,4))
        classes_found = class_counts.index.tolist()
        counts_found = class_counts.values.tolist()

        bars = ax_dist.bar(classes_found,counts_found)
        ax_dist.set_xlabel("Predicted Class")
        ax_dist.set_ylabel("Number of Images")
        ax_dist.set_title("Predicted Class Distribution")

        for bar, count in zip(bars, counts_found):
            ax_dist.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

#                     Confidence distribution

        st.divider()
        st.subheader("Confidence Distribution")

        fig_conf, ax_conf = plt.subplots(figsize=(7,4))
        ax_conf.hist(df["confidence"], bins=20,alpha=0.8)
        ax_conf.axvline(
            x=threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold})"
        )

        ax_conf.set_xlabel("Confidence")
        ax_conf.set_ylabel("Number of Images")
        ax_conf.set_title("Confidence Score Distribution")

        ax_conf.legend()
        plt.tight_layout()
        st.pyplot(fig_conf, use_container_width=True)
        plt.close(fig_conf)

        st.divider()
        st.subheader("Detailed Evidence Log")

        display_cols = ["filename", "predicted_class", "confidence", "flagged"]

        if expected_class != "Unknown / Mixed":
            display_cols.append("is_anomaly")

        display_cols.extend(["cover_prob", "lsb_prob", "wow_prob","s-uniward_prob"])

        st.dataframe(
            df[display_cols],
            use_container_width=True
        )
























