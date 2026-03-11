# 🧠 PathMem: Toward Cognition-Aligned Memory Transformation for Pathology MLLMs

# 🤖 Overview 
<center>
  <img src="https://github.com/jylEcho/PathMem/blob/main/figs/introductionV16.jpg" alt="">
</center>

## Overview of PathMem. WM activates relevant LTM and transforms them into an updated WM for interpretable reasoning

# 📡 Method Overview 

<center>
  <img src="https://github.com/jylEcho/PathMem/blob/main/figs/FrameworkV15.jpg" alt="">
</center>

## Framework of PathMem. A memory-augmented MLLMs for computational pathology that aligns visual, textual, and knowledge graph representations, and adaptively activates LTM for knowledge-grounded reasoning about pathology.


# ⚙️ Step-1. Environment Setup

## 🖥️ 1.1 System Requirements

- **OS:** Linux  
- **Python:** 3.10.18  
- **PyTorch:** 2.7.1 + CUDA 12.6  
- **GPU:** recommended (for other components in the full project)


## 🧪 1.2 Create Conda Environment

```bash
conda create -n pathologykg python=3.10.18
conda activate pathologykg
```

## 🔥 Install PyTorch

```
Install PyTorch with CUDA 12.6:
  pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Verify installation:
  python -c "import torch;print(torch.__version__)"
```

## 📦 Install Dependencies

```
  pip install biopython
  pip install requests
  pip install urllib3
  pip install tqdm
```

# 🧬 Step-2.Pathology Knowledge Graph Construction from PubMed

## 🔍 Overview of KG-Construction

<center>
  <img src="https://github.com/jylEcho/PathMem/blob/main/figs/KGV5.jpg" alt="">
</center>


## ⚙️ 2.1 Configuration

```
Edit the following parameters in config.py
```

### 📧 2.1.1 PubMed API


### 📡 2.1.2 Set your email (required by NCBI):

```
  ENTREZ_EMAIL = "your_email@example.com"
```
  
### 🤖 2.1.3 LLM API

```
  Configure your LLM API:
    API_KEY = "your_api_key"
    API_URL = "https://api.yunwu.ai/v1/chat/completions"
    MODEL_NAME = "gpt-4o"
```

The LLM is used to extract structured pathology information from abstracts.

### 📁 2.1.4 Data Paths

```
BASE_DIR = "/path/to/data"
RAW_DIR = BASE_DIR + "/raw_extractions"
KG_DIR = BASE_DIR + "/kg"
MEMORY_DIR = BASE_DIR + "/memory"
```

## 🔗 2.2 Knowledge Graph Construction Pipeline

### 🔍 2.2.1 PubMed Retrieval

The system automatically queries PubMed using a pathology-focused query:

```
("lung squamous cell carcinoma"[MeSH Terms]
 OR "lung squamous cell carcinoma"[Title/Abstract])
AND
(histopathology OR morphology OR immunohistochemistry OR IHC)
AND hasabstract[text]
```

The pipeline retrieves article PMIDs and downloads their metadata and abstracts.

### 🧠 2.2.2 LLM-Based Information Extraction

Each abstract is processed by an LLM to extract structured pathology knowledge.

The LLM outputs a JSON structure including:

```
  Disease
  Sites
  Histology
  Morphological features
  Biomarkers
  Diagnostic clues
```

Example schema:

```
{
  "disease": {"name": "", "qualifiers": []},
  "sites": {"primary_site": "", "metastatic_sites": []},
  "histology": {"histologic_type": ""},
  "features": {
    "architectural_patterns": [],
    "cellular_features": [],
    "morphologic_features": []
  },
  "biomarkers": {...},
  "diagnostic_clues": [...]
}
```

The LLM extraction strictly follows a predefined schema to ensure structured output. 

### 🧩 2.2.3 Knowledge Graph Construction

Extracted information is converted into triples:

```
(head entity, relation, tail entity)
```

Example:

```
lung squamous cell carcinoma  HAS_IHC_MARKER  p40
lung squamous cell carcinoma  HAS_ARCHITECTURE  keratinization
lung squamous cell carcinoma  HAS_MUTATION  TP53
```

The graph builder also stores additional metadata:
- **PMID**
- **confidence score**
- **evidence span**

Edges are filtered using a confidence threshold.

### 💾 2.2.4 Graph Storage

Two types of graph outputs are generated：

```
triples.tsv
edges.jsonl
```

### ▶️ 2.2.5 Graph Storage

```
python main.py
```

# 🏋️ Step 3.Model training

Run the following script for model training:

```
./WSI_LLAVA/scripts/v1_5/finetune_lora.sh
  --image_folder: path to the extracted feature files (.pt files)
  --data_path: path to the training data (.json files)
  --output_dir: path to save the trained model weights
```

# 🔎 Step 4.Model Inference

Run the following script for model inference:

```
  ./WSI_LLAVA/scripts/wsi-vqa.sh
```


