
## 😊 Facial Emotion Recognition using CNN (FER2013 Dataset)

This project implements a Convolutional Neural Network (CNN) model to classify facial expressions into seven distinct emotions using the FER2013 dataset. The model is designed with regularization and augmentation strategies to ensure robustness and generalizability.

---

### 📁 Dataset

The project uses the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) consisting of 48x48 grayscale images labeled with seven emotions:

* Anger 😠
* Disgust 🤢
* Fear 😨
* Happiness 😄
* Sadness 😢
* Surprise 😲
* Neutral 😐

---

### 🧠 Model Architecture

The CNN model is composed of:

* 3 Convolutional Blocks (64 → 128 → 256 filters)

  * ELU activations
  * He normal initialization
  * Batch Normalization
  * MaxPooling and Dropout
* Dense layer with 128 units
* Output layer with 7 neurons (Softmax)

🔧 **Regularization Techniques**:

* Batch Normalization
* Dropout (0.4–0.6)
* Data Augmentation (Rotation, Shift, Zoom, Flip)

---

### 🚀 Training Details

* **Epochs**: 92
* **Batch Size**: 32
* **Optimizer**: Adam
* **Callbacks**: EarlyStopping & ReduceLROnPlateau
* **Train/Validation Split**: 90/10
* **Input Shape**: (48, 48, 1)

---

### 📊 Performance Report

| Metric                  | Value |
| ----------------------- | ----- |
| **Validation Accuracy** | 68.4% |
| **Validation Loss**     | 0.88  |

#### 🔍 Classification Report:

| Emotion   | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Anger     | 0.61      | 0.65   | 0.63     |
| Disgust   | 0.83      | 0.36   | 0.51     |
| Fear      | 0.58      | 0.40   | 0.47     |
| Happiness | 0.89      | 0.87   | 0.88     |
| Sadness   | 0.59      | 0.53   | 0.56     |
| Surprise  | 0.77      | 0.78   | 0.77     |
| Neutral   | 0.58      | 0.80   | 0.67     |

**Macro Average F1-score**: `0.64`
**Weighted Average F1-score**: `0.68`

---

### 📈 Visualizations

* Emotion class distribution using Seaborn
* Sample grid of expression images
* Training and Validation loss/accuracy curves *(optional: can be added)*

---

### ⚙️ Setup Instructions

1. Clone the repository

   ```bash
   git clone <your-repo-link>
   cd facial-emotion-recognition
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script

   ```bash
   python train.py
   ```

---

### 📌 Future Improvements

* Improve performance on **Disgust** and **Fear** classes via class rebalancing or focal loss
* Experiment with **ResNet-like architectures** or **Vision Transformers (ViTs)**
* Integrate webcam-based real-time emotion detection
* Expand to video emotion recognition tasks

---

### 📚 References

* FER2013 Dataset: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
* Keras Docs: [https://keras.io](https://keras.io)
* CNN Architectures: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
