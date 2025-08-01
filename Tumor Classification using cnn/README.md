
# 🧠 Brain Tumor Detection with MRI Scans

> “Saving lives, one pixel at a time.”  
A deep learning project built to classify brain tumors using MRI images. Powered by both a custom CNN and a Keras model, this project leverages the **Brain MRI Images for Brain Tumor Detection** dataset from Kaggle.

---

## 📁 Dataset Structure

Make sure your directory looks like this after downloading the dataset:

```
Dataset/
└── Data/
    ├── yes/   ← Tumor-positive MRI scans
    └── no/    ← Tumor-negative MRI scans
```

---

## 🛠️ Preprocessing & Augmentation

Prepping your data is key. Here's how to set the stage before running the models:

### 🔹 Preprocessing
Create this folder structure:
```
Preprocessed Data/
├── yes/
└── no/
```
Then run:
```bash
python "Image Preprocessing.py"
```

### 🔹 Augmentation
Prepare these folders:
```
Augmented Data/
├── yes/
└── no/
```
Then execute:
```bash
Augmentation.ipynb
```

*These steps help normalize and expand the dataset to improve model robustness.*

---

## 🚀 Model Implementations

### 🧠 Custom CNN – `Custom_Implementation.ipynb`
Built from scratch, neuron by neuron. Designed with fundamental layers and fine-tuned through experimentation.

- 🎯 **Accuracy:** ~83%  
- 📊 **F1 Score:** 0.83

### ⚙️ Keras Model – `Model_implementation.ipynb`
A cleaner, quicker build using Keras’ Sequential API—great for prototyping and benchmarking.

- 🎯 **Accuracy:** ~85%  
- 📊 **F1 Score:** 0.83–0.84

> 📈 *Both notebooks include training & validation plots for visual insights.*

---

## 🚀 Quickstart

Get things up and running in a flash:

```bash
git clone <your_repo_url>
cd <your_repo_folder>

python -m venv venv
venv\Scripts\activate     # Windows
# OR
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

You’re now ready to preprocess, augment, and train like a pro. 🧪

---

## 👨‍💻 Maintainer

[**Nishanth Devabathini**](https://www.linkedin.com/)  
🎓 B.Tech CSE | Amrita Vishwa Vidyapeetham  
🧠 AI Explorer | 📊 Deep Learning Enthusiast | 💻 Full-stack Tinkerer  

> 📨 Feel free to connect, collaborate, or contribute.  
> ✍️ Raise an issue, star the repo, or just say hi!

---


