# PathMem: Toward Cognition-Aligned Memory Transformation for Pathology MLLMs

## Step 1.Environment Setup

### System Requirements

OS: Linux
Python: 3.10.18
PyTorch: 2.7.1 + CUDA 12.6
GPU recommended (for other components in the full project)

### Create Conda Environment
  conda create -n pathologykg python=3.10.18
  conda activate pathologykg

### Install PyTorch
Install PyTorch with CUDA 12.6:
  pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Verify installation:
  python -c "import torch;print(torch.__version__)"

### Install Dependencies
  pip install biopython
  pip install requests
  pip install urllib3
  pip install tqdm

## Step 2.Pathology Knowledge Graph Construction from PubMed

### Configuration
1、Edit the following parameters in config.py
1.1、PubMed API
1.2、Set your email (required by NCBI):
  ENTREZ_EMAIL = "your_email@example.com"
1.3、LLM API
  Configure your LLM API:
    API_KEY = "your_api_key"
    API_URL = "https://api.yunwu.ai/v1/chat/completions"
    MODEL_NAME = "gpt-4o"




## Step 3.Model training

Run the following script for model training:
./WSI_LLAVA/scripts/v1_5/finetune_lora.sh
  --image_folder: path to the extracted feature files (.pt files)
  --data_path: path to the training data (.json files)
  --output_dir: path to save the trained model weights

## Step 4.Model Inference
Run the following script for model inference:
  ./WSI_LLAVA/scripts/wsi-vqa.sh


