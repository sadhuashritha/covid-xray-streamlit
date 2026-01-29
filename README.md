🦠 COVID-19 X-Ray Detection using Deep Learning (Streamlit App)

This project is a web-based application built using Streamlit and a Convolutional Neural Network (CNN) model to detect COVID-19 from chest X-ray images. The trained deep learning model analyzes uploaded X-ray images and predicts whether the patient is COVID-positive or normal.

🚀 Features

Upload chest X-ray images

Deep Learning based COVID detection

Fast and accurate predictions

User-friendly Streamlit web interface

Real-time result display

🛠 Tech Stack

Python

Streamlit

TensorFlow / Keras

OpenCV

NumPy

Pillow

📁 Project Structure
covid-xray-streamlit/
│
├── app.py                     # Streamlit application file
├── covid_xray_cnn_final.keras # Trained CNN model
├── requirements.txt           # Required Python libraries
├── .gitignore
└── README.md

⚙️ Installation Steps
1️⃣ Clone the Repository
git clone https://github.com/your-username/covid-xray-streamlit.git
cd covid-xray-streamlit

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

5️⃣ Open Browser

Go to:

http://localhost:8501

🧠 Model Details

Model Type: Convolutional Neural Network (CNN)

Input: Chest X-ray Image

Output: COVID Positive / Normal

Format: .keras model file

Framework: TensorFlow + Keras

📸 How It Works

User uploads X-ray image

Image is preprocessed

CNN model predicts the result

Output is displayed on UI

⚠️ Disclaimer

This project is for educational and research purposes only.
It should NOT be used as a medical diagnosis tool.

👨‍💻 Author

Sadhu Ashritha
B.Tech CSE Student
Deep Learning | Python | AI | Full Stack Developer

⭐ Support

If you like this project:

Star ⭐ the repository

Fork 🍴 it

Share with others
