# RainPrediction

# 🌧️ Rainfall Prediction Using Machine Learning

This project demonstrates how machine learning can be used to predict whether it will rain today based on various atmospheric features. Using real-world weather data, we train and evaluate multiple classification models to perform binary prediction of rainfall.

---

## 📂 Dataset

The dataset contains 366 rows and 12 columns, with features like:

- Pressure
- Max/Min/Avg Temperature
- Dew Point
- Humidity
- Cloud Cover
- Sunshine
- Wind Direction
- Wind Speed
- Rainfall (Target variable)

> **Note**: The dataset should be named `Rainfall.csv` and placed in the project directory.

---

## 🛠️ Tools & Libraries Used

- **Python**
- **Pandas** and **NumPy** – data manipulation
- **Matplotlib** and **Seaborn** – data visualization
- **Scikit-learn** – preprocessing, modeling, evaluation
- **XGBoost** – advanced gradient boosting classifier
- **Imbalanced-learn** – handling data imbalance via oversampling

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values and imputed them using the mean.
- Cleaned column names with extra spaces.
- Analyzed distribution of features using histograms and boxplots.
- Visualized correlations using a heatmap.
- Removed highly correlated features (`maxtemp`, `mintemp`) to reduce redundancy.

---

## ⚖️ Data Preprocessing

- Categorical target (`rainfall`: yes/no) converted to binary (1/0).
- Data imbalance handled using **RandomOverSampler**.
- Feature scaling applied using **StandardScaler**.

---

## 🤖 Models Used

Three different classification models were trained and compared:

1. **Logistic Regression**
2. **XGBoost Classifier**
3. **Support Vector Classifier (SVC)**

Each model was evaluated using **ROC AUC Score** on training and validation sets.

---

## ✅ Model Evaluation

| Model                | Training AUC | Validation AUC |
|---------------------|--------------|----------------|
| Logistic Regression | 0.889        | 0.897          |
| XGBoost Classifier  | 0.990        | 0.841          |
| SVC (RBF Kernel)    | 0.903        | 0.886          |

### Final Model: **SVC**

- Confusion matrix and classification report were used to evaluate performance.
- Precision, recall, and F1-score showed strong results, especially for predicting rainy days.

---

## 📈 Results

The model can accurately predict whether rainfall will occur based on input features with ~85% accuracy. While not perfect, it showcases how machine learning can assist in weather prediction.

---

## 📌 Future Improvements

- Hyperparameter tuning for better model performance
- Use of ensemble models
- Incorporate additional weather data like region, month, or historical trends

---

## 👨‍💻 Author

- **Rajesh M** – _Implementation, testing, and GitHub deployment_

---

## 💡 Inspiration

Weather prediction is traditionally dependent on expert meteorologists. This project demonstrates how **Machine Learning** can augment or partially automate such predictions using historical data and smart algorithms.

---

## 📎 License

This project is open-sourced under the [MIT License](LICENSE).

