# What Factors Might Influence the Risk of Getting Diabetes?

## Abstract
Diabetes remains a critical public health issue, particularly in Georgia, where it affects nearly 1 million individuals and places a significant burden on healthcare systems. This study leverages machine learning models, including Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, and Neural Networks, to analyze key predictors of diabetes using the Pima Indians Diabetes Database. By examining physiological, demographic, and genetic factors such as glucose levels, BMI, and age, we identify critical contributors to diabetes risk and assess the performance of various predictive models. Our findings provide insights into diabetes progression and offer actionable knowledge to inform prevention and intervention strategies.

## Introduction


## Data and Setup
The dataset used for this experiment is the Pima Indians Diabetes Database sourced from Kaggle, comprising 768 rows and 9 columns. The dataset includes variables such as the number of pregnancies, plasma glucose concentration, diastolic blood pressure, triceps skin fold thickness, 2-hour serum insulin, body mass index (BMI), diabetes pedigree function (a measure of genetic influence), age, and the outcome variable indicating diabetes diagnosis (0 = No, 1 = Yes). The dataset is imbalanced, with 500 non-diabetic cases (65%) and 268 diabetic cases (35%).

From the pairplot, we observe that Glucose and BMI have a strong visual relationship with Outcome, and features like Insulin and SkinThickness contain outliers. The diagonal histograms also show that Pregnancies and Insulin are highly skewed, indicating potential preprocessing needs. The heatmap further quantifies these relationships. Glucose has the strongest correlation with Outcome (0.49), making it a key predictor. Variables like BMI and Age followed with a strong correlation to Outcome. While some features have notable correlations, others, like DiabetesPedigreeFunction and BloodPressure, show weak associations with Outcome. This analysis guides feature importance evaluation and model refinement.

The experimental setup involves splitting the data into training and testing sets using an 80-20 ratio. Data cleaning was performed by replacing zero values with the mean of each respective feature to handle missing or implausible entries. The models evaluated include Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and a Neural Network, with performance assessed using K-fold cross-validation to ensure robustness. The experiments were executed in a standard computing environment with all necessary libraries and frameworks pre-installed.


## Results
1. Cross-Validation

2. KNN

3. Decision Tree

4. Model Selection

5. Random Forest

We evaluated five machine learning models—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Neural Network—using various performance metrics, including Accuracy, Precision, Recall, F1 Score, ROC AUC, and Cross-Validation (CV) Mean and Standard Deviation.

Based on the results, Random Forest was identified as the best-performing model. It achieved the highest Recall (0.7091), F1 Score (0.6667), ROC AUC (0.8349), and Cross-Validation Mean (0.7753), demonstrating robustness across multiple metrics. While Logistic Regression attained the highest Accuracy (0.7532), this result may be influenced by the imbalance in the outcome variable, which can skew accuracy as a reliable measure.
Given its consistent performance across Recall, ROC AUC, and Cross-Validation metrics, we selected Random Forest as the optimal model for the remainder of our analysis. This model's ability to balance precision and recall makes it particularly suitable for identifying individuals at risk of diabetes.

6. Feature Importance

We utilized the results from our Random Forest model to analyze feature importance, as shown in the graph above. The Mean Decrease in Impurity (MDI) highlights the key predictors for diabetes. The top three most influential features are Glucose, BMI, and Age, which align with existing medical knowledge and intuition.

Glucose is the most significant feature, reflecting its central role in diagnosing diabetes. BMI, a measure of body fat, and Age, a well-known risk factor, further emphasize the importance of physiological and demographic variables in diabetes prediction. Other features, such as Diabetes Pedigree Function and Blood Pressure, contribute to the model but to a lesser extent. These findings reinforce the robustness of our Random Forest model in identifying clinically relevant predictors of diabetes.

## Discussion


## Conclusion
