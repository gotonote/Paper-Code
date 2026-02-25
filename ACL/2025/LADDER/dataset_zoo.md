## 📚 Table of Contents (Datasets Section)

- [📁 Dataset Directory Structure](#-dataset-directory-structure)
- [📥 Dataset Download](#-dataset-download)
  - [✅ Automated Download (Waterbirds & MetaShift)](#-automated-download-waterbirds--metashift)
  - [📎 Manual Download Required](#-manual-download-required)
- [🧪 Preprocessing Mammograms](#preprocessing-mammograms)
- [🗂️ Metadata Files](#metadata-files)


### 📁 Dataset Directory Structure

We follow the directory structure below for the datasets used in this project. The datasets are organized into
subdirectories, each containing the necessary files for training and evaluation.

```bash
data/
 celeba/
    img_align_celeba/
    list_attr_celeba.csv
    list_bbox_celeba.csv
    list_eval_partition.csv
    list_landmarks_align_celeba.csv
    metadata_celeba.csv
    va_metadata_celeba_captioning_blip.csv
    va_metadata_celeba_captioning_GPT.csv
 metashift/
    metadata_metashift.csv
    metadata_metashift_captioning.csv
    te_metadata_metashift_captioning.csv
    va_metadata_metashift_captioning_blip.csv
    va_metadata_metashift_captioning_gpt.csv
    va_metadata_metashift_captioning_GPT.csv
    MetaShift-Cat-Dog-indoor-outdoor/
        train/
        test/
 nih/
    mimic-cxr-chexpert.csv
    nih_processed_v2.csv
 RSNA_Cancer_Detection/
    rsna_w_upmc_concepts_breast_clip.csv
 Vindr/
    vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/
        breast-level_annotations.csv
        finding_annotations.csv
        vindr_detection_v1_folds_abnormal.csv
 waterbirds/
     metadata_waterbirds.csv
     va_metadata_waterbirds_captioning_blip.csv
     va_metadata_waterbirds_captioning_GPT.csv
     waterbird_complete95_forest2water2/
         001.Black_footed_Albatross/
         002.Laysan_Albatross/
         003.Sooty_Albatross/
         ...
         200.Common_Yellowthroat/
         metadata.csv
```

### 📥 Dataset Download

We rely heavily on the [Subpopulation Shift Benchmark (SubpopBench)](https://github.com/YyzHarry/SubpopBench) codebase
for downloading and processing datasets. Necessary compatibility modifications are included in our repo
under `src/codebase/SubpopBench-main`:

#### ✅ Automated Download (Waterbirds & MetaShift)

Use the following command to download the **Waterbirds** and **MetaShift** datasets:

```bash
python ./src/codebase/SubpopBench-main/subpopbench/scripts/download.py \
  --datasets "waterbirds" "metashift" \
  --data_path "Ladder/data/new" \
  --download True
```

#### 📎 Manual Download Required

The following datasets must be downloaded manually from their official sources:

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [NIH ChestX-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [RSNA Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr-Mammo](https://vindr.ai/datasets/mammo)

### Preprocessing Mammograms

Follow the steps
in [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP/tree/main?tab=readme-ov-file#pre-processing-images) codebase to
preprocess the mammograms for RSNA and VinDr Datasets. This step is necessary to convert the dicom images into a png
format compatible with our paper. Also, we uploaded the VinDr png
images [here](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png). If you download the
png images for VinDr from the above link, don't preprocess for the VinDr dataset again.

### Metadata Files (including the annotations and attributes)
The metadata files to train the classifier are provided here:
- [Waterbirds](https://github.com/batmanlab/Ladder/blob/main/data/waterbirds/metadata_waterbirds.csv)
- [MetaShift](https://github.com/batmanlab/Ladder/blob/main/data/metashift/metadata_metashift.csv)
- [CelebA](https://github.com/batmanlab/Ladder/blob/main/data/celeba/metadata_celeba.csv)
- [NIH](https://github.com/batmanlab/Ladder/blob/main/data/nih/nih_processed_v2.csv)
- [RSNA-Mammo](https://github.com/batmanlab/Ladder/blob/main/data/RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv)
- [VinDr-Mammo](https://github.com/batmanlab/Ladder/blob/main/data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv)