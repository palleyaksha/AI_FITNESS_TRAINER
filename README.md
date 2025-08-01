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
## 📦 Dependencies

- **Python 3.8+**
- **PyTorch**
- **torchvision**
- **timm**
- **scikit-learn**
- **matplotlib, seaborn**
- **Flask**
- **MediaPipe**

## 📂 Project Structure
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
## Screenshots
1. Server Launch Screen
   <img width="981" height="544" alt="image" src="https://github.com/user-attachments/assets/53352665-17e0-400c-8fcc-5dad582823d6" />

2. Landing Page 
   <img width="1886" height="950" alt="image" src="https://github.com/user-attachments/assets/49fdb7f2-b7c5-428b-b86c-3050f79e758f" />

3. Home Page
   <img width="1878" height="944" alt="image" src="https://github.com/user-attachments/assets/241d624c-8d66-4cae-8443-7f9422386901" />

4. Upload Video Detection  
   <img width="1888" height="934" alt="image" src="https://github.com/user-attachments/assets/81a2811b-862c-458e-b71e-8d78425dd5c7" />

5. Results for Uploaded Video
   <img width="1863" height="945" alt="image" src="https://github.com/user-attachments/assets/121ac0f8-211a-4544-9ac1-faabb3a1378f" />

6. Live Camera Results
   <img width="1144" height="609" alt="image" src="https://github.com/user-attachments/assets/a361a4ab-f17b-408a-81c6-beb103114292" />
   
7. Workout Progress
   <img width="1068" height="566" alt="image" src="https://github.com/user-attachments/assets/c23c2733-6acd-4cc4-9c89-4427a5cf347d" />

8. History of Workouts
   <img width="1090" height="592" alt="image" src="https://github.com/user-attachments/assets/a65035bd-b450-4a1d-a5ff-dabfa355e4ad" />

9. Feedback section 
   <img width="1179" height="658" alt="image" src="https://github.com/user-attachments/assets/56057d60-0008-4bae-a295-a0a5ab6262c2" />

10. ChatBot
    <img width="641" height="883" alt="image" src="https://github.com/user-attachments/assets/0c10483a-0e5e-44ff-877b-3af86c5abb24" />

## 🎯 Future Enhancements

- Add more exercise classes and fine-tune for complexity.

- Mobile app deployment with TensorFlow Lite / PyTorch Mobile.

- Voice guidance and repetition counting with NLP integration.

## 🙌 Acknowledgements

- BEiT Transformer by Microsoft

- MediaPipe Holistic Model by Google

- PyTorch for deep learning

## 📝 License

This project is licensed under the MIT License.

## 📞 Contact

- Palle Yaksha Reddy - palleyaksha28@gmail.com

- Project Link: https://github.com/palleyaksha/AI_FITNESS_TRAINER



