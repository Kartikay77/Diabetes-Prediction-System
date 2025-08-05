# 💉 Diabetes Prediction System

This project is a machine learning-based diagnostic tool that estimates an individual’s likelihood of having diabetes using key health indicators such as glucose level, BMI, age, and blood pressure. It is built with Python and leverages popular data science libraries like Pandas, NumPy, and scikit-learn.

---

## 🚀 Features

- 🧪 **Exploratory Data Analysis (EDA)** using the PIMA Indian Diabetes dataset  
- 🔄 **Preprocessing Pipeline**: Handles missing values and scales features  
- 🤖 **Model Training** with **Logistic Regression** for binary classification  
- 📊 **Performance Evaluation**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix  
- 🖥️ **CLI-Based Prediction**: Enter patient data manually for real-time prediction  
- 🌐 **Web Deployment** via Streamlit (supports both manual and NLP-based inputs)

---

## 📁 Dataset

- **Source**: [PIMA Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**:
  - `Pregnancies`  
  - `Glucose`  
  - `BloodPressure`  
  - `SkinThickness`  
  - `Insulin`  
  - `BMI`  
  - `DiabetesPedigreeFunction`  
  - `Age`  
  - `Outcome` (target: 0 = No diabetes, 1 = Diabetes)

---

## 🧰 Libraries Used

- Python 3  
- `pandas`, `numpy`  
- `matplotlib`, `seaborn`  
- `scikit-learn`  
- (Optional: `streamlit`, `joblib` for web app deployment)

---

## 🧠 Model

- Algorithm: **Logistic Regression**  
- Train-Test Split: **80:20**  
- Evaluation Metrics:  
  - Accuracy  
  - Precision, Recall, F1-Score  
  - Confusion Matrix (visualized via heatmap)

---

## 📈 Results

- **Accuracy**: ~77%  
- **Confusion Matrix**: Heatmap visualization  
- **Classification Report**: Generated using `sklearn.metrics.classification_report`

---

## 🔧 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Kartikay77/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System

# 2. Install Requirements
pip install -r requirements.txt

3. Run the Notebook
Launch Diabetes_prediction_system.ipynb in Jupyter Notebook or VSCode and run the cells to:
Explore the data
Train and evaluate the model
Use the CLI to predict diabetes manually
4. Try the Web App (Streamlit)
✅ Live App: Click here to try it!
✅ **Live Web App**: [Try it here](https://diabetes-predictor-7nkfu3yuwvfajcqxucytrx.streamlit.app)
🌱 Future Work
Improve model accuracy using ensemble methods like Random Forest or XGBoost
Add explainability using SHAP or LIME
Deploy advanced UI/UX with chatbot integration for patient interaction
Incorporate additional features like smoking history and family medical records

---

Let me know if you'd like a separate `requirements.txt` or deployment instructions for Streamlit as well!



