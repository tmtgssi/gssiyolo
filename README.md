# Geometric and Sentence-level Semantic Information Transfer Learning for YOLO-Based Saffron Stigma Detection

This repository accompanies the project:

"Geometric and Sentence-Level Semantic Information Transfer Learning for YOLO-Based Saffron Stigma Detection"

Our method introduces a novel Geometric and Sentence-Level Semantic Information (GSSI-YOLO) transfer learning strategy, enabling the network to capture rich contextual and structural features. This approach enhances the model's ability to distinguish fine-grained visual patterns in complex backgrounds — a key challenge in saffron detection.

# 📁 Datasets

All datasets used for training and evaluation are located in:
dataset_saffron/

# 🧠 Pretrained Weights

Pretrained YOLO weights with GSSI enhancements can be found at:
smark_picking_saffron/detection_phase/runs/detect/train6/weights/

# ⚙️ Environment Setup

We recommend creating a virtual environment (conda or venv) and installing the required dependencies. Example requirements include:

torch==2.5.0

torchvision==0.20.0

opencv-python

numpy

matplotlib

ultralytics==8.0.190


Make sure to clone the modified Ultralytics YOLO repo if any changes were made to the architecture .

# 🚀 How to Run

✅ 1. Validate Pretrained Model (Inference)

To reproduce our reported results using pretrained weights:

smark_picking_saffron/detection_phase/test.ipynb

🏋️ 2. Train GSSI-YOLO from Scratch

Training involves two main steps:

🔧 Step 1:  Inpainting Phase
This phase prepares weights to be used in YOLO backbone for semantic-level feature learning.

smark_picking_saffron/inpainting_phase/source_multitask/run.txt

🔗 Step 2: Link Semantic Weights into YOLO

Once training is complete, update the path to the semantic weights in:

smark_picking_saffron/detection_phase/ultralytics/nn/modules/block.py

Line 1962, please replace the placeholder path with the actual path to the trained weights. ^^

🏁 Step 3: Train YOLO with GSSI

smark_picking_saffron/detection_phase/train.ipynb

🧬 Generate Perturbed Training Data

To generate perturbed versions of the dataset for robustness training, run:

prepare_pertubated/prepare_permutated.ipynb

# 📝 Citation

If you find this work useful, please cite our paper (to be updated if published).

# 📧 Contact

For any questions or inquiries, please contact:

Minh Trieu Tran

📨 minhtrieu.tran@gssi.it
