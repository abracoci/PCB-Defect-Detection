# PCB Defect Detection — Image Preprocessing and Data Augmentation

## Project Overview

This repository contains image preprocessing and data augmentation tools developed for a university Machine Learning project focused on Printed Circuit Board (PCB) defect detection.

The goal is to prepare PCB images for machine learning experiments through image standardization, quality control, and data augmentation.

## Main Components

### Original Preprocessing
- Reads labeled PCB images and metadata.
- Performs geometric measurements and alignment quality checks.
- Calibrates quality-control thresholds using good PCB images.
- Exports accepted images and a dataset manifest.

### Flexible Preprocessing
- Processes individual PCB photographs or image folders.
- Detects PCB regions and applies perspective correction.
- Supports configurable image size, margins, and debugging.

### Data Augmentation
- Generates additional training images from accepted PCB images.
- Separates training and testing data by board.
- Exports a manifest describing the resulting dataset.

## Installation

Install the required Python libraries:

`pip install -r requirements.txt`

## Usage

**Original Preprocessing**

`python run_pipeline.py --mode original --input "path/to/pcb_dataset" --output "output_original"`

The input dataset must contain `metadata.csv` and a `raw/` directory.

**Flexible Preprocessing**

`python run_pipeline.py --mode flexible --input "path/to/pcb_image.jpg" --output "output_flexible" --margin 0.20 --debug`

**Data Augmentation**

`python augmentation/data_augmentation.py --manifest "output_original/dataset_aligned/manifest.csv" --output "output_augmented" --train-board B01`

Each execution requires a new output directory.

## Experimental Results

| Metric | Result |
|---|---|
| Original PCB images | 38 |
| Successfully exported images | 36 |
| Images rejected by quality control | 2 |
| Augmented training dataset | 209 |
| Testing dataset | 17 |
| Total dataset after augmentation | 226 |

These results represent preprocessing performance and dataset preparation, not machine learning classification accuracy.

## Limitations

- The original preprocessing was tested using two PCB boards, B01 and B02.
- Two images failed quality-control checks.
- Geometric differences were observed between the boards.
- The testing dataset does not contain an accepted `misaligned_capacitor` sample.
- Flexible preprocessing has not been validated across a large variety of PCB designs.
- Augmented images do not represent additional independent physical boards.

## Project Scope

This repository focuses on image preprocessing, quality control, and data augmentation. Model training and defect classification are outside the scope of these components.

## Data Availability

Original PCB photographs are not included. Users must provide their own appropriately structured dataset and have permission to use the images.

## Academic Context

Developed as part of a university Machine Learning team project.