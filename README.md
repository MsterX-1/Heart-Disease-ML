# Heart Disease Machine Learning Project

This project implements a full ML pipeline on the Heart Disease dataset.

## Structure
```
heart-disease-ml/
├── README.md
├── requirements.txt
├── data/
│   └── heart_disease.csv   # (add dataset here)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── scripts/
│   └── train_pipeline.py
├── models/
│   └── final_model.pkl (saved after training)
└── results/
    └── metrics, plots
```

## Setup
1. Clone repo and enter directory.
2. Create venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (or venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```
3. Place dataset in `data/heart_disease.csv`.

## Usage
- Run notebooks step by step in order for exploration.
- Or run the full pipeline script:
  ```bash
  python scripts/train_pipeline.py
  ```

Outputs:
- Cleaned/transformed data in `data/`
- Metrics in `results/`
- Final model in `models/final_model.pkl`
