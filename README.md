# Brain MRI Segmentation

This repository includes the code for training deep learning models used in the Brain MRI Segmentation project. The models are designed to detect FLAIR abnormalities in MRI images.

## Contents

- [Installation](https://github.com/preetham-ganesh/brain-mri-segmentation#installation)
- [Dataset](https://github.com/preetham-ganesh/brain-mri-segmentation#dataset)
- [Usage](https://github.com/preetham-ganesh/brain-mri-segmentation#usage)
- [Model Details](https://github.com/preetham-ganesh/brain-mri-segmentation#model-details)
- [Support](https://github.com/preetham-ganesh/brain-mri-segmentation#support)

## Installation

### Download the repository

```bash
git clone https://github.com/preetham-ganesh/brain-mri-segmentation.git
cd brain-mri-segmentation
```

### Requirements Installation

Requires: [Pip](https://pypi.org/project/pip/)

```bash
pip install --no-cache-dir -r requirements.txt
```

## Dataset

- The data was downloaded from Kaggle - Brain MRI segmentation [[Link]](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
- After downloading the data, 'archive.zip' file should be saved in the following data directory path 'data/raw_data/'.

## Usage

Use the following commands to run the code files in the repo:

Note: All code files should be executed in home directory.

### Preprocess Dataset

```bash
python3 src/preprocess_data.py --dataset_version 1.0.0
```

or

```bash
python3 src/preprocess_data.py -dv 1.0.0
```

### Model Training & Testing

#### FLAIR Abnormality Classification

```bash
python3 src/flair_abnormality_classification/run.py --model_version 1.0.0 --experiment_name flair_abnormality_classification
```

or

```bash
python3 src/flair_abnormality_classification/run.py -mv 1.0.0 -en flair_abnormality_classification
```

#### FLAIR Abnormality Segmentation

```bash
python3 src/flair_abnormality_segmentation/run.py --model_version 1.0.0 --experiment_name flair_abnormality_segmentation
```

or

```bash
python3 src/flair_abnormality_segmentation/run.py -mv 1.0.0 -en flair_abnormality_segmentation
```

### Predict

#### FLAIR Abnormality Classification

```bash
python3 src/flair_mri_classification/predict.py --model_version 1.0.0 --image_file_path <file_path>
```

or

```bash
python3 src/flair_mri_classification/predict.py -mv 1.0.0 --ifp <file_path>
```

#### FLAIR Abnormality Segmentation

```bash
python3 src/flair_mri_segmentation/predict.py --model_version 1.0.0 --image_file_path <file_path>
```

or

```bash
python3 src/flair_mri_segmentation/predict.py -mv 1.0.0 --ifp <file_path>
```

## Model Details

| Model name                       | Model Version | Description                                                                                       | Performance                              | Model URL                                                                                         |
| -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------- |
| FLAIR Abnormality Classification | v1.2.0        | - A CNN model that classifies whether a given Brain MRI image has abnormality.                    | Accuracy: 95.8%                          | [Hugging Face](https://huggingface.co/preethamganesh/bms-flair-abnormality-classification-v1.2.0) |
| FLAIR Abnormality Segmentation   | v1.0.0        | - A U-Net model with MobileNetV2 pretrained on ImageNet as Encoder, and custom layers as decoder. | Dice Coefficient: 75.3% <br/> IoU: 60.9% | [Hugging Face](https://huggingface.co/preethamganesh/bms-flair-abnormality-segmentation-v1.0.0)   |

## Support

For any queries regarding the repository please contact 'preetham.ganesh2021@gmail.com'.
