# Medical Insurance Cost Prediction using Multiple Linear Regression

## Objective
The objective of this project is to estimate the medical insurance charges of customers based on their personal and health-related information.This is achieved by developing a Multiple Linear Regression model to predict these charges.

## Dataset Link
Medical Cost Personal Insurance Dataset
Kaggle Link: https://www.kaggle.com/datasets/mirichoi0218/insurance

## Task 1: Data Understanding
Before developing the model, the dataset was loaded and analyzed to identify the types of features present:
* **Numerical Features:** Age, BMI, Children
* **Categorical Features:** Sex, Smoker, Region
* **Target Variable:** Charges

*(Note: The first five records were successfully loaded and visually verified during the initial script execution to confirm data integrity).*

## Libraries Used
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Methodology
1. **Data Preprocessing:** Checked the dataset for missing values and encoded the categorical variables (sex, smoker, region) to prepare them for the machine learning algorithm.The data was then split into 80% training and 20% testing sets.
2. **Model Development:** Trained a Multiple Linear Regression model on the training data using the specified features (Age, Sex, BMI, Children, Smoker, Region) to predict the target variable (Charges).
3. **Model Evaluation:** The trained model was used to predict charges on the unseen test dataset. Performance was quantified using standard regression metrics, and an Actual vs. Predicted scatter plot was generated to visually assess accuracy.

## Results
* **Mean Absolute Error (MAE):** *4181.19*
* **Mean Squared Error (MSE):** *33596915.85*
* **R-squared Score:** *0.7836*

## Conclusion
The Multiple Linear Regression model successfully identifies the primary factors affecting insurance charges, with smoking status, age, and BMI acting as the most significant drivers of medical costs. Key findings indicate that as these risk factors increase, the predicted insurance charges rise accordingly.However, a notable limitation of Linear Regression for this problem is its core assumption of strictly linear relationships.In reality, health insurance costs often scale non-linearly; for example, the compounding effect of a high BMI combined with smoking triggers exponentially higher charges. A basic linear model struggles to perfectly capture these complex, synergistic interactions without the manual addition of polynomial or interaction terms, which leads to occasional underpredictions for extreme high-cost outliers.