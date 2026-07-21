# Medical Insurance Cost Prediction using Multiple Linear Regression

## Objective
The purpose of this project is to predict the medical insurance charges for customers using their personal and health-related information. A Multiple Linear Regression model is used to accomplish this prediction task.

## Dataset Link
Medical Cost Personal Insurance Dataset
Kaggle Link: https://www.kaggle.com/datasets/mirichoi0218/insurance

## Task 1: Data Understanding
The dataset was loaded and analyzed to understand the types of features present:
* **Numerical Features:** Age, BMI, Children
* **Categorical Features:** Sex, Smoker, Region
* **Target Variable:** Charges

*(Note: The first five records were loaded and verified to confirm data is correct).*

## Libraries Used
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Methodology
1. **Data Preprocessing:** The dataset was checked for missing values. Categorical variables (sex, smoker, region) were encoded into numerical format. The data was then divided into 80% training and 20% testing sets.
2. **Model Development:** A Multiple Linear Regression model was trained on the training data using the features Age, Sex, BMI, Children, Smoker, and Region to predict Charges.
3. **Model Evaluation:** The trained model was used to make predictions on the test dataset. The performance was measured using standard regression metrics, and an Actual vs. Predicted scatter plot was created to visualize the model's accuracy.

## Results
* **Mean Absolute Error (MAE):** *4181.19*
* **Mean Squared Error (MSE):** *33596915.85*
* **R-squared Score:** *0.7836*

## Conclusion
The Multiple Linear Regression model successfully identified the main factors that affect insurance charges. The most important factors are smoking status, age, and BMI. As these factors increase, the predicted insurance charges also increase. However, there is a limitation to using Linear Regression for this problem. The model assumes all relationships are linear. In reality, insurance costs do not always follow straight-line patterns. For example, when a customer has both a high BMI and is a smoker, the insurance charges increase much faster than expected. The basic linear model cannot capture these complex interactions without adding extra features. This sometimes results in lower predictions for customers with very high insurance costs.