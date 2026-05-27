# 🩺 Breast Cancer Diagnostic System

A Machine Learning-based web application built with Streamlit that predicts whether a breast tumor is **Benign** or **Malignant** using clinical measurement data.

---

## 📌 Project Overview

This project uses a **Decision Tree Classifier** trained on breast cancer diagnostic data to help analyze tumor characteristics and predict cancer severity.

The application provides a simple and interactive user interface where users can enter clinical measurements and instantly receive prediction results.

---

## 🚀 Features

- Interactive Streamlit web interface
- Real-time tumor diagnosis prediction
- Decision Tree Machine Learning model
- User-friendly clinical input form
- Instant result visualization
- Lightweight and easy to run

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## 📂 Project Structure

```text
breast-cancer-diagnostic-system/
│
├── app.py
├── data.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 📊 Dataset Information

The dataset contains breast cancer clinical measurements such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Symmetry
- Fractal Dimension

Target Labels:

- `M` → Malignant (Cancerous)
- `B` → Benign (Non-Cancerous)

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-diagnostic-system.git
```

---

### 2️⃣ Navigate to Project Folder

```bash
cd breast-cancer-diagnostic-system
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🖥️ Application Workflow

1. Enter patient clinical measurements
2. Click **Analyze Results**
3. System predicts:
   - Malignant
   - Benign

---

## 📸 Screenshots

Add application screenshots inside:

```text
screenshots/
```

Example:

- Home Page
- Prediction Result
- Input Form

---

## 📈 Machine Learning Model

Model Used:

```text
Decision Tree Classifier
```

Reason for selection:

- Fast training
- Easy interpretability
- Good performance on structured datasets

---

## 📌 Future Improvements

- Add multiple ML algorithms
- Improve UI design
- Add model accuracy visualization
- Deploy on Streamlit Cloud
- Add patient report export feature

---

## 👨‍💻 Author

Tapabrata Sau

---

## 📄 License

This project is licensed under the MIT License.
