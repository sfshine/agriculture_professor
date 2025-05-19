# Agricultural Disease Detection

This project implements a deep learning model for agricultural disease detection.

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── data/              # Data processing modules
│   │   ├── dataset.py     # Dataset loading and processing
│   │   ├── filter_data.py # Data filtering utilities
│   │   └── analyze_labels.py # Label analysis tools
│   ├── models/            # Model related code
│   │   ├── train_model.py # Model training script
│   │   └── test_model.py  # Model testing script
│   ├── utils/             # Utility functions
│   └── scripts/           # Executable scripts
├── AgriculturalDisease_trainingset/    # Training dataset
├── AgriculturalDisease_validationset/  # Validation dataset
├── agricultural_disease_model.pth      # Trained model weights
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your dataset in the appropriate directories:
   - Training images in `AgriculturalDisease_trainingset/`
   - Validation images in `AgriculturalDisease_validationset/`

## Usage

1. Train the model:
```bash
python src/models/train_model.py
```

2. Test the model:
```bash
python src/models/test_model.py
``` 