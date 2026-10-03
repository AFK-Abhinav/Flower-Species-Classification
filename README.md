# 🌼 FlowerNet — Flower Species Classification with MobileNetV2

A transfer-learning image classifier that identifies five flower species — **daisy, dandelion, roses, sunflowers, tulips** — from photos, built with TensorFlow/Keras and a frozen **MobileNetV2** backbone pretrained on ImageNet.

---

## 📊 Results

| Metric | Score |
|---|---|
| **Final Validation Accuracy** | **90.19%** |
| **Final Validation Loss** | 0.4726 |
| Training Epochs | 15 (best weights restored from epoch 14) |

### Per-Class Accuracy

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Daisy | 107 | 120 | 89.2% |
| Dandelion | 147 | 159 | 92.5% |
| Roses | 114 | 131 | 87.0% |
| Sunflowers | 126 | 138 | 91.3% |
| Tulips | 168 | 186 | 90.3% |

---

## 🧠 Model Architecture

- **Backbone:** MobileNetV2 (`include_top=False`, frozen, ImageNet weights) — 2,257,984 params
- **Head:** GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(5, Softmax)
- **Input size:** 128×128×3
- **Data augmentation:** Random horizontal flip, rotation (±10%), zoom (±10%), contrast (±10%)
- **Trainable params:** 329,221 | **Non-trainable params:** 2,257,984

```
Input (128,128,3)
   → Data Augmentation
   → MobileNetV2 Preprocessing
   → MobileNetV2 Base (frozen)
   → GlobalAveragePooling2D
   → Dense(256, relu)
   → Dropout(0.4)
   → Dense(5, softmax)
```

---

## 📁 Dataset

- **Source:** [`tf_flowers`](https://www.tensorflow.org/datasets/catalog/tf_flowers) via TensorFlow Datasets (auto-downloaded, no login/API key required)
- **Classes (5):** dandelion, daisy, tulips, sunflowers, roses
- **Split:** 80% train (2,936 images) / 20% validation (734 images)

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 (initial) |
| Loss | Categorical Crossentropy (label smoothing 0.05) |
| Batch Size | 16 |
| Epochs | 15 (with Early Stopping) |
| Callbacks | `EarlyStopping` (patience=4), `ReduceLROnPlateau` (factor=0.5, patience=2) |

---

## 🚀 Getting Started

### Requirements
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```

### Run
Open the notebook (`.ipynb`) in Jupyter or Google Colab and run all cells sequentially. The dataset downloads automatically — no manual setup or credentials needed.

### Inference on a new image
```python
predicted_class, confidence = predict_landmark("path/to/your/flower.jpg")
```
This returns the top predicted class along with a confidence score, and displays a bar chart of the top-3 predictions.

---

## 📈 Training Curves

Training and validation accuracy/loss curves are saved automatically to:
```
/content/training_curves.png
```

---

## 🗂️ Project Structure

```
├── Landmark_Detection.ipynb     # Main notebook (data pipeline, model, training, eval, inference)
├── training_curves.png          # Accuracy/loss plots (generated on run)
└── README.md
```

---

## 🔧 Possible Improvements

- Fine-tune the top layers of MobileNetV2 (unfreeze last N layers) for a potential accuracy boost
- Increase input resolution (e.g., 224×224) to better match MobileNetV2's native input size
- Add more aggressive augmentation or mixup to reduce confusion between visually similar classes (e.g., roses vs. tulips)
- Export to TFLite for on-device/mobile inference

---

## 📄 License

Add your preferred license here (e.g., MIT).
