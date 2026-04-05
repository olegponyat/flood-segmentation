# Flood Segmentation with Machine Learning

## Overview

This project leverages the **WorldFloods Dataset** to train a deep learning model for detecting and segmenting flooded areas in satellite imagery. The model classifies each pixel into three categories: **Land** (0), **Water** (1), and **Cloud** (2).

The project implements a U-Net architecture optimized for satellite imagery with custom loss functions—both Focal Loss and Dice Loss with class weighting—to handle class imbalance between land, water, and cloud pixels. Built with PyTorch Lightning, the framework provides streamlined training with integrated callbacks and checkpoint management. The model delivers comprehensive performance metrics using IoU, Recall, and Dice Loss evaluation, while the inference pipeline enables water polygon extraction and geographic visualization with GeoPandas for seamless GIS integration.

> [!IMPORTANT]
> This project requires **30+ GB of storage** for the full WorldFloods dataset and works best with an NVIDIA GPU. **DO NOT ATTEMPT TO RUN WITHOUT EITHER OF THESE REQUIREMENTS.**

### Dependencies
```
gdown                  # For downloading WorldFloods dataset
ml4floods              # Machine learning for flood detection
torch                  # PyTorch deep learning framework
pytorch-lightning      # Training framework
torchvision            # Computer vision utilities
geopandas              # Geographic data processing
rasterio               # Raster image I/O
matplotlib             # Visualization
numpy                  # Numerical computing
pandas                 # Data analysis
```

### Installation

```bash
# Install required packages
pip install gdown ml4floods torch pytorch-lightning geopandas matplotlib pandas numpy

# For GPU support, ensure CUDA is installed
# PyTorch with CUDA: https://pytorch.org/get-started/locally/
```


## Custom Loss Functions

The project implements a hybrid loss function combining two complementary approaches:

- **Focal Loss**: Addresses class imbalance by down-weighting easy examples and focusing on hard negatives
- **Dice Loss**: Direct optimization of the Intersection over Union (IoU) metric, crucial for segmentation tasks
- A weighted combination of both losses is then used for overall class balance

> [!NOTE]
> The extreme class weight for water (10M) compensates for the severe class imbalance in satellite imagery where water pixels are much rarer than land pixels. This weighting scheme ensures the model learns to detect water effectively despite its scarcity in training data.

## Usage

### Quick Start

```python
# 1. Load configuration
from ml4floods.models.config_setup import get_default_config
config = get_default_config('models/worldfloods_template.json')
config.experiment_name = 'my_experiment'

# 2. Load dataset
from ml4floods.models.dataset_setup import get_dataset
dataset = get_dataset(config.data_params)

# 3. Create and train model
from ml4floods.models.model_setup import get_model
model = get_model(config.model_params)
trainer.fit(model, dataset)

# 4. Run inference
inference_function = get_model_inference_function(model, config)
outputs = inference_function(satellite_image)
```

## Satellite Data

**Sentinel-2 Bands Used:**
- **Blue** (B2), **Green** (B3), **Red** (B4) - RGB composite
- **NIR** (B8), **SWIR1** (B11) - Vegetation and water indices

**Spatial Resolution:** 10-20m per pixel

## Output & Postprocessing

- Thresholded probability maps
- Vectorized geometries for GIS integration
- Overlays on RGB and SWIR composites (not used)
- Shapefile or GeoJSON format for downstream analysis

## References

- **ml4floods**: https://github.com/torrelmo/ml4floods
- **PyTorch Lightning**: https://lightning.ai/
- **Sentinel-2**: https://earth.esa.int/eoportal/missions/sentinel-2
- **WorldFloods Dataset**: https://www.drivendata.org/competitions/71/worldfloods/

---

> [!NOTE]
> This project is optimized for Google Colab but can run locally on machines with NVIDIA GPUs and sufficient storage for the dataset (~30+ GB with models). Colab provides free GPU access, making it ideal for initial experimentation.
