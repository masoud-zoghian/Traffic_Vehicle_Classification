# 🚗 Traffic Vehicle Classification

This repository documents two parts/experimental versions of a traffic vehicle image-classification project. **The data structure and experimental goals of these two parts are not identical**, so their results should be interpreted separately.

---

# Part 1 — Baseline CNN and Ablation Experiments

## 🎯 Goal

Build a baseline CNN for vehicle-image classification and investigate the effects of architectural and training changes, keeping each experiment in its own folder.

## 📁 Data Decisions

Images for this part were collected from four training sources:

- `DATA/dataset/train`
- `DATA/dataset/clean`
- `DATA/datasetv2_TrainClean/train`
- `DATA/datasetv2_TrainClean/clean`

The final version of this part used **7 classes**:

`ambulance`, `autobus`, `kamyun`, `minibus`, `savari`, `taxi`, `vanet`

Key decisions:

- `neysan` was merged into `vanet`.
- `kamyunet` was merged into `kamyun`.
- Duplicate files were checked using MD5. Sixteen duplicate groups were identified; after deduplication, **3,868 unique images** remained.
- The test set was kept separate and was not used for model selection or hyperparameter tuning.
- Data was split into training and validation sets using a stratified split with `random_state=42`.
- The training set contained **3,094** images and the validation set contained **774** images. No overlap was found between the two sets.
- Images were resized to `224×224` and normalized using the commonly used ImageNet mean and standard deviation.

## 🧪 Folders and Experiments

| Folder | Main purpose/change | Best Validation Accuracy |
|---|---|---:|
| `04_dataset_transforms` | Dataset definition, class mapping, and base transforms | — |
| `05_dataloader` | Creating batches and train/validation DataLoaders | — |
| `06_training` | Baseline CNN, `CrossEntropyLoss`, and Adam | **79.98%** |
| `07_augmentation` | `RandomHorizontalFlip` and `RandomRotation(10°)` | 68.99% |
| `08_dropout` | Adding `Dropout(p=0.5)` | 83.72% |
| `09_pooling` | Replacing Max Pooling with Average Pooling | **85.27%** |
| `10_weight_decay` | Adam with `weight_decay=0.0001` | **85.14%** |
| `11_scheduler` | `StepLR(step_size=3, gamma=0.1)` | 82.17% |
| `12_balanced_batches` | `WeightedRandomSampler` for more balanced sampling | 82.68% |
| `13_loss_comparison` | Comparing Cross-Entropy and BCE | CE: 81.91% / BCE: 85.40% |

## ⚙️ Base Configuration

- Image size: `224×224`
- `batch_size=32`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Base loss: `CrossEntropyLoss`
- Evaluation metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrix
- The best checkpoint was selected and saved based on validation performance.

## 💡 Key Takeaways from Part 1

- Under the tested settings, **Average Pooling** and **Weight Decay** performed better than the baseline.
- Because class counts were imbalanced, Accuracy alone was not sufficient; macro metrics and the Confusion Matrix were also examined.
- `seed=42` and deterministic settings were used to improve reproducibility.

---

# Part 2 — Data Restructuring, ResNet18, and Cascade

## 🔄 Complete Data-Structure Change

At this stage, **the overall project data structure was changed** and the data for this part was organized under `newData/`. The goal was to separate training sources, known-class test data, out-of-distribution (OOD) data, and split files.

The documented structure was:

```text
newData/
├── dataset/
│   ├── train/
│   └── clean/
├── datasetv2_TrainClean/
│   ├── train/
│   └── clean/
├── dataTest/
│   ├── test/
│   └── test_neysan_subset/
├── neysan_ood/
└── resnet_data_splits/
```

Trained models and threshold settings were saved in `saved_models/`.

## 🗂️ Main Data Decisions

In this version, the **8 original classes were retained**:

`ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, `vanet`

Unlike Part 1, `kamyun` and `kamyunet` were not merged in the main dataset. They were mapped to a shared `truck_merged` group only in the first-stage Coarse Classifier of the Cascade; the second model then distinguishes between the two original classes.

- Four training sources were collected into a central inventory.
- Initial inventory: **3,313 images**.
- Duplicate detection was performed on image content: each image was converted to RGB and resized to `64×64` before hashing.
- **11 duplicate groups**, covering 22 records, were identified; no label conflicts were found.
- Within each duplicate group, the version whose path contained `clean` was retained. The physical files were not deleted; only duplicate records were excluded from the DataFrame.
- After deduplication, **3,302 unique images** remained.
- A stratified split was created using `VAL_RATIO=0.20` and `SPLIT_SEED=42`.
- Hash overlap between training and validation was zero.
- Final manifest: `newData/resnet_data_splits/inventory_train_val_split.csv`
- `dataTest/test` was reserved for known-class testing and excluded from training.
- `neysan_ood` was used as an unseen, out-of-distribution class and was not included in model training.

## 🧠 Architecture: ResNet18 + Cascade

Instead of the simple CNN, **ResNet18 pretrained on ImageNet** was used. The final layer was replaced to produce the required number of project outputs.

The pipeline has two stages:

1. **Coarse ResNet18:** predicts 7 groups. At this stage, `kamyun` and `kamyunet` are mapped to `truck_merged`.
2. **Cascade ResNet18:** only when the first-stage prediction is `truck_merged`, the image is passed to the second model to classify it as `kamyun` or `kamyunet`.

The Coarse model was trained on 3,302 images. The Cascade model used only the two classes `kamyun` and `kamyunet`: 758 training images and 191 validation images.

## 🧪 ResNet18 Experiment Results

| Experiment | Recorded result |
|---|---:|
| Feature Extraction — freeze the backbone and train the head | 91.11% Validation Accuracy |
| Full Fine-Tuning | 98.80% Validation Accuracy |
| Fine-Tuning with Weighted Cross-Entropy | **99.10% Validation Accuracy** |
| Balanced Batch Sampler | 98.80% Validation Accuracy |
| Two-class Cascade for `kamyun/kamyunet` | 93.19% Validation Accuracy |

**Model selection:** Weighted Loss achieved the best validation accuracy in the Coarse experiments, and this model was used in the pipeline and threshold stage. During fine-tuning, separate learning rates were used for the backbone and the head so that pretrained weights would be updated more conservatively.

## ✅ Main Test on Known Classes — Phase 1

The two-stage model was evaluated on `dataTest/test`:

- Number of images: **365**
- Incorrect predictions: **8**
- Accuracy: **97.81%**
- Error rate: **2.19%**
- Macro F1: **97.22%**

This result is for the known-class Phase 1 test and is distinct from the validation accuracy reported during training.

## 🕵️ Threshold, Neysan OOD, and `needs_review`

This stage aimed to indicate whether a prediction should be reviewed by a human, in addition to returning a predicted class.

- Total images in `neysan_ood`: **554**
- 277 images were used to calibrate the threshold.
- The other 277 images were held out for the Phase 2 test to keep threshold evaluation separate.
- Selected threshold: **0.99**
- During calibration, this threshold flagged about **29.24%** of Neysan images as below the threshold; the false-flag rate on known data was about **10.09%**.
- In Phase 2, about **27.44%** of Neysan images were marked for review; the review rate on known data was **13.42%**.
- If `confidence < 0.99`, `needs_review=True` is set.

For the Cascade pipeline, the final confidence was calculated as the product of the Coarse-stage confidence and the Cascade-stage confidence.

### ⚠️ Important Limitation of Confidence

**High confidence does not guarantee a correct prediction or prove that an image belongs to the training distribution.** Error analysis found two known-class errors with very high confidence:

- An `ambulance` with confidence `0.999620` was incorrectly predicted as `vanet`.
- A `kamyun` with confidence `0.999821` was incorrectly predicted as `kamyunet`.

Many Neysan images were also assigned to known classes with high confidence. Therefore, the threshold and `needs_review` are only supporting mechanisms for flagging potentially suspicious samples. This approach is **not a complete OOD detector** and cannot conclusively determine whether an image belongs to an unknown class.

## 📌 Part 2 Summary

- ResNet18 with transfer learning achieved higher validation performance than the simple CNN.
- Weighted Loss achieved the best Coarse-model validation accuracy: **99.10%**.
- The Cascade enabled a second-stage distinction between the similar `kamyun` and `kamyunet` classes.
- The pipeline achieved **97.81%** accuracy on the known-class test.
- The `0.99` threshold flagged some Neysan images, but many OOD samples were still classified with high confidence.
- Confidence should be interpreted alongside error analysis and a review policy, not as a guarantee of correctness.

---

## 📊 Keep the Results Separate

| Item | Part 1 | Part 2 |
|---|---|---|
| Model type | Baseline CNN | ResNet18 + Cascade |
| Number of original dataset classes | 7 | 8 |
| Handling of `kamyun/kamyunet` | Merged in the final dataset | Kept separate; merged temporarily only for Coarse |
| Neysan OOD data | Included in the final class mapping for this part | Separate `neysan_ood` set, excluded from training |
| Key best result | Average Pooling: 85.27% validation accuracy | Weighted Loss: 99.10% validation accuracy |
| Final known-class test | Not reported for this part | Phase 1: 97.81% on 365 images |
| OOD / threshold analysis | Not reported for this part | Threshold = 0.99; limited OOD detection ability |

> **Note:** The results from the two parts are not directly comparable because their data structures, class counts, architectures, and evaluation procedures differ.

## 🧰 Important Folders and Notebooks

### Part 1

- `04_dataset_transforms`
- `05_dataloader`
- `06_training`
- `07_augmentation`
- `08_dropout`
- `09_pooling`
- `10_weight_decay`
- `11_scheduler`
- `12_balanced_batches`
- `13_loss_comparison`

### Part 2

- `newData/dataset`
- `newData/datasetv2_TrainClean`
- `newData/dataTest/test`
- `newData/neysan_ood`
- `newData/resnet_data_splits`
- `1_data_audit.ipynb`
- `2_3_dataset_transforms.ipynb`
- `4_5_resnet.ipynb`
- `6_threshold.ipynb`
- `saved_models/` — model weights and threshold settings


---

## 📚 Project Reports

Detailed explanations of the project workflow, implementation, and design decisions are available in these PDF files:

- `Project-Report.pdf`
- `Project-Report-part2.pdf`

## 🚀 Running the Project

To run the project, use `predict.py`. You can use either of the following methods.

### Option 1: Run from the command line

```bash
python predict.py --image path/to/image.jpg
```

Replace `path/to/image.jpg` with the path to the image you want to classify.

### Option 2: Use it as an importable Python function

```python
from predict import predict_single_image

result = predict_single_image("path/to/image.jpg")
```

The prediction result is returned in the `result` variable.
