Day 4 — Machine Learning: Facility Hygiene Risk Prediction

1. Project Overview

This project uses Machine Learning to predict facility hygiene risk based on facility conditions. It demonstrates the complete Machine Learning workflow, including data cleaning, preprocessing, model training, evaluation, and prediction.

The project is developed using Python and Scikit-learn.

2. Objectives

- Understand basic Machine Learning concepts.
- Clean and prepare a dataset.
- Select features and target labels.
- Split data into training and testing sets.
- Train and compare two Machine Learning algorithms.
- Evaluate models using classification metrics.
- Predict hygiene risk for a sample facility.

3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- VS Code

4. Project Structure

day-04/
├── dataset/
│   ├── facilities.csv
│   └── cleaned_facilities.csv
├── preprocessing/
│   └── clean_data.py
├── models/
│   ├── train_model.py
│   ├── logistic_regression.joblib
│   └── decision_tree.joblib
├── evaluation/
│   ├── evaluate_model.py
│   └── model_comparison.json
├── predictions/
│   ├── predict_risk.py
│   └── facility_predictions.csv
└── README.md

5. Dataset Description

The dataset contains sample facility information.

Feature| Description
facility_id| Unique facility identifier
cleanliness_score| Cleanliness score from 0 to 10
odor_score| Odor severity score from 0 to 10
waste_level| Low, Medium, or High
footfall| Number of visitors
complaints| Number of complaints
hours_since_cleaning| Hours since the last cleaning
hygiene_risk| Example target label: Low, Medium, or High

The dataset contains 20 sample records created for educational practice. These are not real inspection records.

6. Data Preprocessing

The preprocessing program:

- Loads the original CSV dataset.
- Checks missing values and duplicate rows.
- Converts numerical columns into numeric format.
- Removes rows with missing required values.
- Checks score ranges and invalid negative values.
- Saves the cleaned dataset as "cleaned_facilities.csv".

7. Machine Learning Algorithms

Logistic Regression

Logistic Regression is a classification algorithm used to predict categories. In this project, it predicts Low, Medium, or High hygiene risk.

Decision Tree Classifier

A Decision Tree makes predictions using a sequence of feature-based decisions. It is also used to classify facility hygiene risk.

Both algorithms use the same training and testing split for comparison.

8. Model Evaluation

The models are evaluated using:

- Accuracy: Proportion of correct predictions.
- Precision: How many predicted cases of a class were correct.
- Recall: How many actual cases of a class were identified correctly.
- F1 Score: Balance between precision and recall.
- Confusion Matrix: Compares actual and predicted classes.

The training script saves the model comparison results in "evaluation/model_comparison.json".

Actual performance results should be added here after running the evaluation script.

9. Making Predictions

The prediction program loads the trained Logistic Regression model and predicts the hygiene risk for a sample facility.

The prediction is saved in "predictions/facility_predictions.csv".

10. How to Run the Project

Step 1: Install Python Libraries

Open the terminal in the project folder and run:

python -m pip install pandas numpy matplotlib scikit-learn joblib

Step 2: Clean the Dataset

python .\preprocessing\clean_data.py

Step 3: Train the Models

python .\models\train_model.py

Step 4: Evaluate the Models

python .\evaluation\evaluate_model.py

Step 5: Predict Hygiene Risk

python .\predictions\predict_risk.py

Run these commands from the "day-04" project folder and complete each step before moving to the next.

11. Problems Encountered

Potential challenges during development include:

- Handling missing and invalid data.
- Encoding categorical features such as waste level.
- Preparing numerical features for training.
- Avoiding errors caused by missing files or columns.
- Comparing model performance on a small dataset.

Document the actual problems encountered and their solutions after running the programs.

12. Limitations

- The dataset is small and uses example-generated risk labels.
- The model may learn patterns from those example labels rather than real hygiene outcomes.
- The measured performance does not establish real-world reliability.
- More representative, verified facility inspection data is needed for meaningful predictions.

13. Future Improvements

- Collect a larger dataset with verified hygiene-risk labels.
- Add more relevant facility and inspection features.
- Compare additional algorithms, such as Random Forest.
- Use cross-validation and tune model parameters.
- Add charts and a dashboard for interactive predictions.
- Monitor model performance on new, verified data.

14. Conclusion

This project demonstrates a basic end-to-end Machine Learning workflow for facility hygiene risk classification. It covers data preprocessing, feature preparation, training two classification algorithms, evaluation, and prediction. The project provides a foundation for further learning and improvement using real-world data.
