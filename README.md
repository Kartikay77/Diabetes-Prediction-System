Diabetes Prediction System

This project implements a machine learning-based prediction system that estimates the likelihood of diabetes in individuals using key health parameters such as glucose level, BMI, age, and blood pressure. The system is built using Python and popular data science libraries like Pandas, NumPy, and scikit-learn.

🚀 Features

Exploratory Data Analysis (EDA) on PIMA Indian Diabetes dataset
Preprocessing including handling missing values and feature scaling
Model training using Logistic Regression
Performance evaluation using confusion matrix, accuracy, precision, recall, and F1-score
Simple CLI interface for predictions
📁 Dataset

The model is trained on the PIMA Indians Diabetes Database, which is widely used for binary classification problems related to diabetes prediction. It contains the following features:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (target)
You can download the dataset from Kaggle if not already included.

📊 Libraries Used

Python 3
pandas
numpy
matplotlib
seaborn
scikit-learn
🧠 Model

The system uses Logistic Regression to perform binary classification. The data is split into training and testing sets using an 80:20 ratio, and the model is evaluated using standard classification metrics.

📈 Results

The model achieves good predictive performance based on the selected metrics. Example output includes:

Accuracy: ~77%
Confusion Matrix: Displayed via heatmap
Precision, Recall, F1-Score: Reported using classification_report from sklearn
🔧 Usage

Clone the repository:
git clone https://github.com/Kartikay77/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
Install required libraries:
pip install -r requirements.txt
Run the notebook:
Open Diabetes_prediction_system.ipynb in Jupyter Notebook or VSCode.
Run prediction manually:
At the end of the notebook, enter user input to get a diabetes prediction based on the trained model.
📝 Future Work

Improve accuracy using ensemble methods like Random Forest or XGBoost
Deploy as a web app using Flask or Streamlit
Add SHAP or LIME explanations for interpretability
