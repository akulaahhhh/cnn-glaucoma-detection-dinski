# 👁️ Glaucoma Detection using CNN — Team Dinski

> **IDSC 2026 — International Data Science Challenge**
> *Theme: Mathematics for Hope in Healthcare*
> Competition website: [https://idsc2026.github.io/](https://idsc2026.github.io/)

A deep learning pipeline for detecting **Glaucomatous Optic Neuropathy (GON)** from retinal fundus images using the [Hillel Yaffe Glaucoma Dataset (HYGD)](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/) from PhysioNet. The pipeline includes exploratory data analysis, image preprocessing with CLAHE enhancement, a custom CNN model with data augmentation, and Grad-CAM interpretability.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [Prerequisites](#-prerequisites)
4. [Setup Instructions](#-setup-instructions)
5. [Dataset Download](#-dataset-download)
6. [Pipeline — Step-by-Step Execution](#-pipeline--step-by-step-execution)
7. [Methodology](#-methodology)
8. [Model Architecture](#-model-architecture)
9. [Interpretability — Grad-CAM](#-interpretability--grad-cam)
10. [Ethical Considerations & Limitations](#-ethical-considerations--limitations)
11. [Citations](#-citations)

---

## 🔬 Project Overview

| Item              | Detail                                                                              |
| ----------------- | ----------------------------------------------------------------------------------- |
| **Task**          | Binary classification — GON+ (glaucoma) vs GON- (healthy)                          |
| **Dataset**       | HYGD — Hillel Yaffe Glaucoma Dataset (retinal fundus images + `Labels.csv`)         |
| **Model**         | Custom CNN (3 Conv blocks → Dense → Sigmoid)                                       |
| **Interpretability** | Grad-CAM heatmaps on the last convolutional layer                               |
| **Framework**     | TensorFlow / Keras                                                                  |
| **Language**      | Python 3.10+                                                                        |

---

## 📂 Repository Structure

```
glaucoma_detection_dinski/
│
├── 01_eda_data_overview.py          # EDA — view basic dataset structure
├── 02_eda_data_quality.py           # EDA — check missing values, duplicates, label distribution
├── 03_eda_quality_score_analysis.py # EDA — analyze image quality scores by label
├── 04_eda_image_analysis.py         # EDA — verify image files exist & check dimensions
├── 05_preprocessing_data_cleaning.py# Clean data: drop cols, encode labels, filter quality
├── 06_preprocessing_image.py        # Image preprocessing: resize + CLAHE enhancement
├── 07_data_splitting.py             # Patient-level train/test split (80/20)
├── 08_check_test_train_dataset.py   # Verify train/test split balance
├── 09_build_model.py                # Build, train, and save the CNN model
├── 10_evaluate_model.py             # Evaluate: confusion matrix + classification report
├── 11_evaluate_gradcam.py           # Generate Grad-CAM heatmap visualizations
│
├── Images/                          # ⬇️ Raw fundus images (downloaded from PhysioNet)
├── Images_Processed/                # Auto-generated: preprocessed images (224×224 + CLAHE)
├── Labels.csv                       # ⬇️ Original labels file (downloaded from PhysioNet)
├── Labels_cleaned.csv               # Auto-generated: cleaned labels
├── train_dataset.csv                # Auto-generated: training split metadata
├── test_dataset.csv                 # Auto-generated: testing split metadata
├── glaucoma_model_dinski.h5         # Auto-generated: trained model weights
├── cam.jpg                          # Auto-generated: Grad-CAM output image
│
├── requirements.txt                 # Python dependencies (pip)
├── notes.md                         # Internal progress notes
├── .gitignore                       # Git ignore rules
└── README.md                        # ← You are here
```

> **Legend:** Items marked with ⬇️ must be downloaded manually (see [Dataset Download](#-dataset-download)). Items marked as *Auto-generated* are created by running the pipeline scripts.

---

## ✅ Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- **Git** — [Download Git](https://git-scm.com/downloads)
- **A PhysioNet account** — required to download the HYGD dataset ([Register here](https://physionet.org/register/))

### Hardware Recommendations

| Component | Minimum       | Recommended        |
| --------- | ------------- | ------------------ |
| RAM       | 8 GB          | 16 GB              |
| GPU       | Not required  | NVIDIA GPU w/ CUDA |
| Storage   | 2 GB free     | 5 GB free          |

> 💡 **Tip:** Training works on CPU but is significantly faster with a GPU. Google Colab (free tier) can be used as an alternative.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/akulaahhhh/cnn-glaucoma-detection-dinski.git
cd glaucoma_detection_dinski
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including TensorFlow, Keras, OpenCV, scikit-learn, Pandas, Matplotlib, Seaborn, and Pillow.

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import cv2; print('OpenCV', cv2.__version__)"
```

You should see version numbers printed without errors.

---

## 📥 Dataset Download

The dataset is the **Hillel Yaffe Glaucoma Dataset (HYGD)** hosted on PhysioNet.

### Step-by-Step Download

1. **Go to the dataset page:**
   [https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)

2. **Sign in** to your PhysioNet account (create one if you don't have one).

3. **Sign the Data Use Agreement** — scroll down on the dataset page, read the agreement, and click to accept.

4. **Download the files.** You need two things:
   - `Labels.csv` — the labels and metadata file
   - `Images/` — the folder containing all fundus image `.jpg` files

5. **Place the downloaded files** into the project root so your folder looks like this:

   ```
   glaucoma_detection_dinski/
   ├── Images/
   │   ├── image_001.jpg
   │   ├── image_002.jpg
   │   └── ... (all fundus images)
   ├── Labels.csv
   └── ... (other project files)
   ```

> ⚠️ **Important:** The `Images/` folder and `Labels.csv` are listed in `.gitignore` and are **not included** in this repository. You must download them yourself from PhysioNet.

---

## 🚀 Pipeline — Step-by-Step Execution

Run the scripts **in numbered order** from the project root directory. Each script is self-contained and produces outputs needed by the next step.

> ⚠️ **Make sure your virtual environment is activated** and you are in the project root directory before running any script.

---

### Phase 1 — Exploratory Data Analysis (EDA)

These scripts help you understand the dataset before any processing. They print outputs to the terminal and show visualizations.

```bash
# Step 1: View basic dataset structure (shape, columns, first rows, statistics)
python 01_eda_data_overview.py

# Step 2: Check data quality (missing values, duplicates, label distribution)
python 02_eda_data_quality.py

# Step 3: Analyze image quality scores per label (shows distribution plot)
python 03_eda_quality_score_analysis.py

# Step 4: Verify image files exist and check image dimensions
python 04_eda_image_analysis.py
```

**What to expect:**
- Steps 1–2 & 4: Text output in the terminal
- Step 3: A histogram plot window will open showing quality score distributions for GON+ vs GON-

---

### Phase 2 — Data Preprocessing

These scripts clean the data and prepare the images for modeling.

```bash
# Step 5: Clean the label data
#   → Removes 'Unnamed: 4' column
#   → Encodes labels: GON+ → 1, GON- → 0
#   → Filters out images with Quality Score < 3.8
#   → Saves: Labels_cleaned.csv
python 05_preprocessing_data_cleaning.py

# Step 6: Preprocess all images
#   → Resizes every image to 224×224 pixels
#   → Applies CLAHE contrast enhancement
#   → Saves processed images to Images_Processed/
python 06_preprocessing_image.py
```

**What to expect:**
- `Labels_cleaned.csv` is created in the project root
- `Images_Processed/` folder is created with all enhanced images

---

### Phase 3 — Data Splitting

```bash
# Step 7: Split data into train (80%) and test (20%) sets
#   → Uses PATIENT-LEVEL splitting (no data leakage)
#   → Saves: train_dataset.csv, test_dataset.csv
python 07_data_splitting.py

# Step 8: Verify the train/test split
#   → Prints image counts and GON+/GON- balance for each split
python 08_check_test_train_dataset.py
```

**What to expect:**
- `train_dataset.csv` and `test_dataset.csv` are created
- Terminal shows image counts and class distribution percentages

> 💡 **Why patient-level splitting?** To prevent data leakage. The same patient can have multiple images — if one ends up in training and another in testing, the model may "cheat" by memorizing patient-specific features rather than learning generalizable patterns.

---

### Phase 4 — Model Training

```bash
# Step 9: Build and train the CNN model
#   → Loads preprocessed images + labels
#   → Applies real-time data augmentation (flip + rotation)
#   → Trains with EarlyStopping (patience=10)
#   → Saves: glaucoma_model_dinski.h5
python 09_build_model.py
```

**What to expect:**
- Training progress printed epoch by epoch
- Training stops early when validation loss plateaus
- `glaucoma_model_dinski.h5` saved (~65 MB)
- **Estimated time:** 5–15 min (GPU) · 30–60 min (CPU)

---

### Phase 5 — Model Evaluation

```bash
# Step 10: Evaluate the trained model
#   → Generates predictions on the test set
#   → Displays confusion matrix + classification report heatmaps
python 10_evaluate_model.py
```

**What to expect:**
- Confusion matrix printed to terminal + visualized as heatmap
- Classification report with precision, recall, F1-score for GON+ and GON-

---

### Phase 6 — Interpretability (Grad-CAM)

```bash
# Step 11: Generate Grad-CAM heatmap
#   → Picks a random test image
#   → Produces a visual heatmap showing which regions the model focused on
#   → Saves: cam.jpg
python 11_evaluate_gradcam.py
```

**What to expect:**
- Model prediction score printed to terminal
- Raw heatmap displayed
- Superimposed Grad-CAM image displayed and saved as `cam.jpg`

---

### 📋 Quick Reference — Run All Steps

If you want to run the entire pipeline at once:

```bash
python 01_eda_data_overview.py
python 02_eda_data_quality.py
python 03_eda_quality_score_analysis.py
python 04_eda_image_analysis.py
python 05_preprocessing_data_cleaning.py
python 06_preprocessing_image.py
python 07_data_splitting.py
python 08_check_test_train_dataset.py
python 09_build_model.py
python 10_evaluate_model.py
python 11_evaluate_gradcam.py
```

> ⚠️ **Note:** Steps 3, 10, and 11 open matplotlib plot windows. Close each plot window to let the script finish before moving to the next step.

---

## 🧠 Methodology

### 1. Data Cleaning
- Removed the extraneous `Unnamed: 4` column from the raw CSV
- Encoded labels into numeric format (`GON+` → 1, `GON-` → 0)
- Filtered out low-quality images with `Quality Score < 3.8` to ensure model learns from clear retinal features

### 2. Image Preprocessing
- **Resize:** All images resized to **224 × 224 pixels** for uniform model input
- **CLAHE Enhancement:** Applied Contrast Limited Adaptive Histogram Equalization in the LAB color space (L-channel only) to improve contrast while preserving color information
  - `clipLimit = 2.0`, `tileGridSize = (8, 8)`

### 3. Data Splitting Strategy
- **Patient-level 80/20 split** using `train_test_split` with `random_state=42`
- All images from the same patient are kept in the same split → **prevents data leakage**

### 4. Data Augmentation (Real-time)
- Random horizontal & vertical flips
- Random rotation (±20%)
- Applied during training only (not saved to disk)

### 5. Training Strategy
- Optimizer: **Adam**
- Loss: **Binary Crossentropy**
- Epochs: up to **100** (with EarlyStopping)
- EarlyStopping: monitors `val_loss`, patience = **10** epochs, restores best weights
- Validation: **20%** of training data held out as validation set

---

## 🏗️ Model Architecture

```
┌────────────────────────────────────────┐
│          Input (224 × 224 × 3)         │
├────────────────────────────────────────┤
│     Data Augmentation (Flip + Rot)     │
├────────────────────────────────────────┤
│   Conv2D(32, 3×3, ReLU) + MaxPool2D   │
├────────────────────────────────────────┤
│   Conv2D(64, 3×3, ReLU) + MaxPool2D   │
├────────────────────────────────────────┤
│  Conv2D(128, 3×3, ReLU) + MaxPool2D   │  ← Grad-CAM target layer
├────────────────────────────────────────┤
│              Flatten                   │
├────────────────────────────────────────┤
│          Dense(64, ReLU)               │
├────────────────────────────────────────┤
│           Dropout(0.5)                 │
├────────────────────────────────────────┤
│         Dense(1, Sigmoid)              │  ← Output: probability of GON+
└────────────────────────────────────────┘
```

---

## 🔍 Interpretability — Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to visualize which regions of the retinal image influenced the model's prediction.

**How it works:**
1. Forward-pass the image through the model
2. Compute the gradient of the predicted class score with respect to the last convolutional layer (`conv2d_2`)
3. Global-average-pool the gradients to get per-channel importance weights
4. Compute a weighted sum of the feature maps → raw heatmap
5. Apply ReLU and normalize → final heatmap
6. Overlay the heatmap on the original image for visual interpretation

**Clinical relevance:** The heatmaps show whether the model is focusing on the **optic disc region**, which is the anatomically correct area for GON diagnosis. This provides transparency and builds trust in the model's reasoning.

---

## ⚖️ Ethical Considerations & Limitations

### Limitations
- **Small dataset size** — the HYGD dataset is relatively small, which limits generalization
- **Class imbalance** — GON+ samples may be underrepresented, leading to bias toward the majority class
- **Single-source data** — all images come from one hospital (Hillel Yaffe), which may not represent diverse populations
- **Quality threshold** — filtering out images below quality 3.8 may exclude edge-case images that a robust model should handle

### Ethical Considerations
- **Not a diagnostic tool** — This model is a research prototype and should NOT be used for clinical diagnosis without proper regulatory approval
- **Bias awareness** — The model is trained on a single demographic/geographic source. Performance may vary across different populations
- **Data privacy** — The dataset is de-identified and accessed through proper PhysioNet agreements
- **Transparency** — Grad-CAM heatmaps are provided to enable clinicians to verify the model's reasoning before trusting any prediction
- **Data leakage prevention** — Patient-level splitting ensures no data leaks between train and test sets

### Responsible Deployment Plan
- The model should be used as a **screening aid** to assist — not replace — ophthalmologists
- Any deployment would require validation on external datasets from diverse populations
- Regular model retraining would be necessary as new data becomes available

---

## 📚 Citations

### Dataset Citation (Required)

> Shrot, S., Bregman-Amitai, O., & Defined in PhysioNet.
> **Hillel Yaffe Glaucoma Dataset (HYGD).**
> PhysioNet. Available at: [https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)

*(Please check the exact citation string at the bottom of the [dataset's PhysioNet page](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/) for the most up-to-date version.)*

### PhysioNet Platform Citation (Required)

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
> **PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.**
> *Circulation*, 101(23), e215–e220.

---

## 📜 License

This project was created for the **IDSC 2026 Data Science Competition**. The dataset is governed by the PhysioNet Data Use Agreement. Please refer to the [PhysioNet terms](https://physionet.org/) for dataset usage rights.

---

<p align="center">
  <b>Team Dinski</b> · IDSC 2026 · Mathematics for Hope in Healthcare
</p>
