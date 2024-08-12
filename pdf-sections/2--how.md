### 2. **How the Model Works**

---

**Basic Explanation**

The model used in this report is an **XGBoostClassifier**, which is a type of machine learning model specifically designed for classification tasks. The primary purpose of this model is to predict outcomes based on patterns found in historical data. In this case, it predicts possible outcomes of eviction cases by analyzing the factors you provided.

**What is XGBoost?**
XGBoost stands for **eXtreme Gradient Boosting**. It’s an advanced implementation of gradient boosting, a machine learning technique that builds a strong prediction model by combining the strengths of several simpler models. Think of it as a team of decision-makers, where each one improves upon the mistakes of the previous one, leading to a more accurate final decision.

**How Does the Model Make Predictions?**
The XGBoostClassifier works by considering multiple factors (also called features or inputs) related to eviction cases. It evaluates these factors based on patterns it has learned from thousands of previous cases and then predicts the likelihood of various outcomes for your case.

The model was trained on a large dataset of eviction cases from Ontario, Canada, allowing it to identify which factors are most likely to influence different outcomes. When you input your data into the Housing Law Insight Dashboard, the model uses this trained knowledge to predict what might happen in your case.

**Key Inputs (Columns):**

The model uses a variety of inputs to make its predictions. To learn more about this aspect of the model training, refer to the **Feature Weights** section of this report.

**Decision-Making Process**

The model processes the information you provided and assigns probabilities to different possible outcomes. These probabilities indicate how likely each outcome is, based on the patterns the model has learned from past cases.

For example, if the model assigns a 70% probability to an eviction outcome, this means that, based on the historical data, broadly speaking, 7 out of 10 cases with similar circumstances resulted in eviction.

By understanding the factors that influence these predictions, you can gain insight into what aspects of your case might be most significant.