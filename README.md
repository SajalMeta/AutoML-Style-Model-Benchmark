# 📊 AutoML Model Comparison Dashboard

> **A streamlined Machine Learning benchmarking tool aimed at automating model selection and performance evaluation.**

![Project Banner](https://via.placeholder.com/1000x300?text=AutoML+Model+Comparison+Dashboard+Banner)

## 📝 Short Summary
This project is an automated ML pipeline that accepts a dataset, preprocesses the data, trains 5 different classifiers (Logistic Regression, KNN, Random Forest, XGBoost, MLP), and generates a comparative performance report via an interactive Streamlit dashboard.

---

## ❓ Problem Statement
In many data science projects, selecting the right baseline model involves repetitive boilerplate code: loading data, scaling features, splitting train/test sets, and running multiple training loops. This manual process is time-consuming and prone to inconsistencies.

**The Goal**: Build a standardized, reusable system that automates this workflow, allowing engineers to focus on feature engineering and model tuning rather than infrastructure.

## 💡 Solution Approach
I designed a modular Python application with a clear separation of concerns:
1.  **Data Layer**: Automatic handling of missing values, label encoding for categorical targets, and standard scaling.
2.  **Model Factory**: A design pattern to extendably instantiate models (Logistic, KNN, Random Forest, XGBoost, Neural Networks).
3.  **Evaluation Engine**: A robust training loop that captures Accuracy, Precision, Recall, and F1-score.
4.  **Interactive UI**: A Streamlit frontend to democratize access to the insights, making it easy to upload data and visualize results instantly.

---

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: MLP Classifier (Scikit-learn)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit

## 🏗️ Project Structure
```bash
ML/AutoML Style Model Benchmark/
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_manager.py     # Data ingestion and preprocessing pipeline
│   ├── model_factory.py    # Factory pattern for model initialization
│   └── evaluator.py        # Core training and metrics evaluation logic
└── README.md               # Project documentation
```

---

## 📊 Dataset Description
The system is designed to be **dataset-agnostic** for classification tasks.
- **Input**: Supports generic CSV files.
- **Format**: Features can be numerical or categorical; the target variable is user-selectable.
- **Built-in Samples**: Includes Iris, Wine, and Breast Cancer datasets for quick demonstration purposes.

## 🤖 Model Pipeline
The system evaluates the following algorithms to cover different learning styles:
1.  **Logistic Regression**: Linear baseline.
2.  **K-Nearest Neighbors (KNN)**: Distance-based classification.
3.  **Random Forest**: Ensemble bagging method for robustness.
4.  **XGBoost**: Gradient boosting for high performance.
5.  **MLP Classifier**: Multi-Layer Perceptron (Neural Network) for non-linear relationships.

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "ML/AutoML Style Model Benchmark"
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 📈 Results & Visualizations
The dashboard provides:
- **Metric Table**: A sorted leaderboard of models based on Accuracy (customizable to Precision/Recall).
- **Comparative Charts**: Bar charts comparing model performance across all metrics.
- **Confusion Matrix**: (Planned) Visual breakdown of true positives vs. false positives.

### Screenshot Placeholder
![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Streamlit+Dashboard+Screenshot)
*(Replace this with an actual screenshot of your app running)*

---

## 🔮 Future Improvements
- **Hyperparameter Tuning**: Integrate GridSearchCV or Optuna for automated tuning.
- **Dockerization**: Containerize the application for easier deployment.
- **Regression Support**: Extend the pipeline to handle regression problems.
- **Model Download**: Allow users to download the best-performing trained model as a `.pkl` file.

---

## 📄 Key Engineering Metrics (Resume Bullets)
*Add these to your resume or portfolio description:*

- **Built automated ML benchmarking pipeline**: Engineered a modular system to process datasets and train 5 distinct classifiers, reducing experimental turnaround time.
- **Compared multiple classifiers**: Implemented a unified evaluation interface to calculate Accuracy, Precision, Recall, and F1-score across Logistic Regression, Random Forest, XGBoost, and Deep Learning models.
- **Generated performance report**: Developed an interactive dashboard using Streamlit to visualize model performance, facilitating data-driven model selection.

---
*Created by Sajal Paik | https://github.com/SajalMeta
