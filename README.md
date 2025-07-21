# MedCLIP RSNA Binary Classification

This repository contains code and scripts for training and evaluating a MedCLIP-based model on the RSNA dataset for binary classification of medical images (e.g., detecting presence/absence of a condition in chest X-rays).

NOTE: This repository was modified to use both [MedCLIP](https://arxiv.org/pdf/2210.10163) method and finetune on RSNA classification task. 

---

## Repository Structure

```
vlm-seminar/
│
├── MedCLIP/
│   ├── examples/
│   │   └── run_medclip_pretrain.py      # Main training & evaluation script
│   ├── dataset/
│   │   └── ...                          # Custom dataset classes
│   ├── modeling_medclip.py              # MedCLIP model definitions
│   ├── losses.py                        # Loss functions
│   ├── evaluator.py                     # Evaluation utilities
│   ├── prompts.py                       # Prompt generation
│   └── constants.py                     # Constants and mappings
│
├── trainer.py                           # Training loop logic
│
├── local_data/
│   ├── train_balanced.csv               # Training data CSV
│   ├── val_balanced.csv                 # Validation data CSV
│   ├── test_balanced.csv                # Test data CSV
│   └── ...                              # (CSV: columns = path, label)
│
├── checkpoints/
│   └── rsna_binary_classification_SU/
│       └── final_model.bin              # Saved model weights
│
└── README.md
```

---

## Data Preparation

### Download the Chexpert Dataset
You can download the dataset from Kaggle using the following link: [Chexpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert). Alternatively, you can download it directly via command line:

```bash
mkdir ~/datasets/chexpert
cd ~/datasets/chexpert
kaggle datasets download ashery/chexpert
unzip chexpert-v10-small.zip -d ./
```
Preprocess using the `rsna.ipynb` notebook in "preprocessing" folder. 

- Place your CSV files (`train_balanced.csv`, `val_balanced.csv`, `test_balanced.csv`) in the `local_data/` directory.
- Each CSV should have at least two columns: `path` (path to DICOM image) and `label` (0 or 1).

---

## Training & Evaluation

The main script is an adapted version of  [`MedCLIP/examples/run_medclip_pretrain.py`](MedCLIP/examples/run_medclip_pretrain.py).

The pretrained checkpoints for MedCLIP were downloaded directly from the official [`MedCLIP repository`](https://github.com/RyanWangZf/MedCLIP/tree/main/medclip)

### **How it works:**

1. **Data Loading:**  
   Loads and preprocesses DICOM images using a custom dataset class. Only 10% of the data is used for faster debugging (can be adjusted).

2. **Model Setup:**  
   Initializes a MedCLIP model with a Vision Transformer (ViT) backbone and a binary classifier.

3. **Training:**  
   Trains the model on the training set, evaluates on the validation set, and saves checkpoints.

4. **Testing:**  
   After training, loads the best model and evaluates on the test set. Prints confusion matrix, accuracy, recall, and test loss.

### **To run:**

```bash
cd MedCLIP/examples
python run_medclip_pretrain.py
```

- The script is set to run on **CPU** by default.  
- Model checkpoints are saved in `checkpoints/rsna_binary_classification_SU/`.

---

## Key Files

- **MedCLIP/examples/run_medclip_pretrain.py**  
  Main script for training and evaluating the MedCLIP model.

- **MedCLIP/modeling_medclip.py**  
  Model architectures for MedCLIP and classifier.

- **MedCLIP/dataset/**  
  Custom dataset classes for loading and preprocessing data.

- **trainer.py**  
  Training loop and checkpointing logic.

- **MedCLIP/evaluator.py**  
  Evaluation metrics and utilities.

---

## Results

After running the script, you will see output like:

```
######### Final Test Results #########
Confusion Matrix:
TP: ..., TN: ..., FP: ..., FN: ...
Accuracy: 0.xxxx
Recall: 0.xxxx
Test Loss: 0.xxxx
```

---

## Customization

- **Data Subsampling:**  
  By default, only 10% of the data is used for quick debugging. Adjust in `RSNASuperviseImageDataset` if needed.

- **Device:**  
  The script uses CPU. To use GPU, change `device = torch.device("cpu")` to `torch.device("cuda")` and adjust environment variables.

- **Batch Size, Epochs, etc.:**  
  Modify the `train_config` dictionary in the script.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas, numpy, scikit-learn
- pydicom

Install dependencies with:

```bash
pip install torch torchvision pandas numpy scikit-learn pydicom Pillow
```

---

## Citation

If you use this code, please cite the original MedCLIP paper and this repository.

---

## License

This project is for research and educational purposes.

---

**Contact:**  
For questions, please open an issue or contact the repository maintainer.




# 🎉 End & Enjoy




