# 🧠 AI Fitness Trainer 🏋️‍♀️

An AI-powered real-time fitness assistant that uses computer vision and deep learning to detect, classify, and evaluate user workout movements. Built with Python, PyTorch, Flask, and MediaPipe.

## 🔍 Project Overview

This project integrates a hybrid deep learning model (BEiT + BiLSTM) to analyze fitness exercise videos, classify workout types, and provide real-time feedback to users. The application includes a drag-and-drop web interface and supports pose detection for accurate movement tracking.

## 📁 Dataset

- Custom dataset consisting of **video recordings** of 4 key exercises.
- Videos were **preprocessed using MediaPipe Holistic model** for pose detection.
- Frames were extracted and **labeled based on the exercise type**.
- Class balancing was performed through oversampling to ensure model fairness.

## 🧠 Model Architecture

- **BEiT (Bidirectional Encoder Representation from Image Transformers)** used for spatial feature extraction.
- **BiLSTM (Bidirectional Long Short-Term Memory)** applied for learning temporal dependencies in sequential image data.
- **Dropout and Fully Connected Layers** used for regularization and final classification.

## 🔄 Methodology

1. **Data Collection & Processing**: Videos collected, divided into frames, and preprocessed (resized, augmented, normalized).
2. **Model Training**:
   - BEiT + BiLSTM hybrid architecture
   - Training with class balancing and augmentation
   - Evaluation with accuracy, precision, recall, and F1-score
3. **Real-Time Inference**: Flask backend processes image input and returns predictions with confidence scores.

## 📊 Performance

- **Accuracy**: 94.2%
- **Precision**: 93.8%
- **Recall**: 94.5%
- **F1-Score**: 94.1%

Evaluation was supported by confusion matrix, loss/accuracy curves, and class-wise metrics.

## 🚀 Deployment

- Built using **Flask** for serving the model through a web interface.
- Drag-and-drop support for user video input.
- Responsive result display with confidence scores.

## 🔧 Installation

```bash
git clone https://github.com/palleyaksha/AI_FITNESS_TRAINER.git
cd AI_FITNESS_TRAINER
pip install -r requirements.txt
python app.py
```
📦 Dependencies

Python 3.8+
PyTorch
torchvision
timm
scikit-learn
matplotlib, seaborn
Flask
MediaPipe

📂 Project Structure
```
AI_FITNESS_TRAINER/
├── app.py                   # Flask app
├── models/                  # Trained model weights
├── templates/               # HTML templates
├── static/                  # CSS, JS, and uploads
├── utils/
│   ├── model.py             # Model architecture
│   └── preprocessing.py     # Image preprocessing
├── dataset/                 # (Optional) Custom dataset
└── README.md
```
##🎯 Future Enhancements

Add more exercise classes and fine-tune for complexity.

Mobile app deployment with TensorFlow Lite / PyTorch Mobile.

Voice guidance and repetition counting with NLP integration.

##🙌 Acknowledgements

BEiT Transformer by Microsoft

MediaPipe Holistic Model by Google

PyTorch for deep learning

##📝 License

This project is licensed under the MIT License.

##📞 Contact

Palle Yaksha Reddy - palleyaksha28@gmail.com

Project Link: https://github.com/palleyaksha/AI_FITNESS_TRAINER



