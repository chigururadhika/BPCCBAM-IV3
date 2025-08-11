# 🌿 BPCCBAM-IV3 Bell pepper Plant Leaf Disease Classification 

The proposed BPCCBAM model  is developed by using Channel Attention Mechanism (CAM), Bit Directions Adaptive Spatial Attention Mechanism (BDASAM) and Positional Attention Mechanism (PAM) for  effeicntly identifying descriminative features for leaf disease regions.  It is integrated with  InceptionV3 for  bellpepper diesease classifcation 

## 🚀 Features

- **Multi-class disease classification** with high accuracy
- **Real-time prediction** on single leaf images
- **Flexible training pipeline** with customizable parameters
- **Pre-trained model** ready for immediate use
- **Easy-to-use command-line interface for image prediction**
- **Comprehensive evaluation metrics**

## 📂 Project Structure

```



plant-leaf-disease-classifier/
├── train.py                 # Training script with model evaluation
├── test.py                  # Prediction and testing utilities
├── model.py                 # model architecture definition
├── data_loader.py           # Dataset loading and preprocessing
├── utils.py                 # Helper functions and utilities
├── requirements.txt         # Python dependencies
├── trained_model.keras      # Pre-trained model weights
└── README.md                # This file
```


## Model Components and Visualizations

### 1. Structure of PAM (Model architecture will be uploaded in Future)
![Structure of PAM](utils/attention3.png)

---

### 2. Bit Directions Adaptive Spatial Attention Mechanism
![Bit Directions Adaptive Spatial Attention Mechanism](utils/attention1.png)

---

### 3. Channel Attention Mechanism
![Channel Attention Mechanism](utils/attention2.png)

---

### 4. BDASAM in InceptionV3 Model
![BDASAM in InceptionV3 Model](utils/model_architecture.png)

---
## 🔧 Installation

### Prerequisites
- Python 3.8 
- GPU (optional, but recommended for training)

### Setup

1. **Clone the repository**
```bash
git __repo_link__  ["replace repo link"]
cd path_folder_downloade
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset** (optional - if not using your own)
```bash

unzip DATASET.zip -d ../dataset/
```

## 📊 Dataset Requirements

Organize your dataset in the following structure:

```
dataset/

├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── ...


**Supported image formats:** JPG, JPEG, PNG  
**Recommended image size:** 299x299 pixels  
**Minimum images per class:** 100+ for good performance

## 🚀 Usage

### 1️⃣ Training a New Model

```bash
# Train from scratch set RETRAIN_MODEL flag in train.py True
python train.py 

```

### 2️⃣ Evaluating the Model

```bash
# Evaluate on test dataset
python test.py

#Predict on a single image:

python test.py --image path/to/your/image.jpg 

#Use custom weights:

python test.py --weights my_model.keras --image leaf.jpg 


```

## 📋 Requirements

```

opencv-python==4.8.0.76
pandas==2.0.3
pillow==10.4.0
scikit-learn==1.3.2
seaborn==0.13.2
tensorflow==2.13.0
tqdm==4.66.1
matplotlib
```


## 📚 References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/adilmubashirchaudhry/plant-village-dataset)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@chigururadhika](https://github.com/chigururadhika)
- Email: chigururadhikaphd@gmail.com


## 🙏 Acknowledgments

- Thanks to the PlantVillage team for providing the dataset
- Inspired by research in agricultural AI applications
- Built with TensorFlow and the amazing open-source community

## 📊 Project Stats

![Python version](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow version](https://img.shields.io/badge/tensorflow-2.8%2B-orange)

---

