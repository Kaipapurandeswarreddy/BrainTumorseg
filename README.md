# 🧠 BraTS 21: AI-Powered Brain Tumor Segmentation

A state-of-the-art medical imaging application that performs 3D volumetric segmentation of brain tumors from MRI scans. Built with **MONAI**, **Gradio**, and **Google Gemini AI**, this tool provides radiologists and researchers with precise tumor boundaries, volume calculations, and automated medical insights.

![BraTS UI](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![AI](https://img.shields.io/badge/AI-Gemini%20Flash-blue?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-MONAI%20%7C%20PyTorch-red?style=flat-square)

## ✨ Features

- **3D Segmentation**: Uses a pre-trained **SegResNet** model to identify:
  - 🔴 **Necrotic/Tumor Core (TC)**
  - 🟢 **Whole Tumor (WT)**
  - 🟡 **Enhancing Tumor (ET)**
- **Interactive Visualization**: Navigate through 155+ MRI slices with a smooth, responsive slider.
- **AI Medical Analysis**: Generates professional clinical summaries using **Google Gemini 2.0/1.5 Flash** (via `google.genai` SDK).
- **Volume Calculation**: Automatically calculates tumor volumes in cm³.
- **Drag & Drop**: Supports NIfTI files (`.nii.gz`) for T1, T1ce, T2, and FLAIR modalities.
- **Production-Grade UI**: Clean, medical-themed interface with dark mode support.

## 🚀 Quick Start (Google Colab)

The easiest way to run this app is on Google Colab (Free GPU recommended).

1. **Upload Files**:
   - Upload `app.py`, `inference.py`, and `requirements.txt` to your Colab workspace.
   - Upload your BraTS MRI data (4 files per patient) and your model checkpoint (`model.pt`).

2. **Install Dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   !python app.py
   ```
4. Click the **public URL** (e.g., `https://xxxx.gradio.live`) to open the app.

## 🛠️ Local Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended for faster inference)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brats-segmentation-app.git
   cd brats-segmentation-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## 📂 File Structure

```
├── app.py                # Main Gradio application (Frontend + AI Logic)
├── inference.py          # Backend inference logic (MONAI + PyTorch)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🤖 AI Integration (Google Gemini)

This app uses the **Google GenAI SDK** to generate medical insights.
- **Input**: Calculated tumor volumes (WT, TC, ET).
- **Model**: Automatically selects the best available model (`gemini-2.0-flash-exp`, `gemini-1.5-flash`).
- **Output**: A concise, 150-word clinical summary for neurologists/radiologists.

**Note**: You will need a valid Google Gemini API Key to use the explanation feature.

## 📦 Dependencies

- `gradio`: Web interface
- `monai`: Medical Deep Learning framework
- `torch`: PyTorch backend
- `numpy`, `matplotlib`: Data processing and visualization
- `nibabel`: NIfTI file handling
- `google-genai`: Google Gemini API client

## 📜 License

This project is intended for **research and educational purposes only**. It is not a certified medical device and should not be used for clinical diagnosis without proper validation.
