### 4. **Model Performance**

---

**Accuracy and Metrics**

Understanding how well the model performs is crucial for interpreting its predictions. In this section, we’ll explain the key metrics used to evaluate the model's performance, which can help you assess the reliability of the predictions.

**Model Performance Metrics**

- **Accuracy:** This measures how often the model correctly predicts the outcome. For example, if the model has an accuracy of 80%, it means that in 8 out of 10 cases, the model’s prediction matched the actual outcome.
  
- **Precision:** Precision indicates how many of the cases predicted as a certain outcome (e.g., eviction) actually ended up that way. High precision means that when the model predicts an eviction, it’s often correct.
  
- **Recall:** Recall, also known as sensitivity, measures how well the model identifies all actual cases of a certain outcome. High recall means that the model successfully identifies most cases that should result in that outcome.
  
- **F1-Score:** The F1-Score is the balance between precision and recall. It’s particularly useful when you want to strike a balance between the two, ensuring that the model is both accurate in its predictions and comprehensive in identifying all relevant cases.

**Performance Summary**

The Housing Law Insight Dashboard uses a model that has been thoroughly evaluated to ensure it provides reliable predictions. Below is a summary of the model’s performance:

- **Accuracy:** 80%
- **Precision for Eviction:** 87%
- **Recall for Eviction:** 93%
- **F1-Score for Eviction:** 90%

These metrics indicate that the model is quite reliable, especially in predicting eviction outcomes. However, no model is perfect, and these numbers also reflect the inherent uncertainty in predicting legal outcomes.

**Baseline Comparison**

To understand how well the model performs, it’s useful to compare it to a baseline. The baseline represents the accuracy you would expect if you made predictions based purely on the most common outcomes in the dataset, without any advanced analysis.

- **Baseline Accuracy:** For example, if the most common outcome in eviction cases is “Full Eviction,” and it occurs 62% of the time, a model that predicts “Full Eviction” for every case would have a baseline accuracy of 62%.

The fact that the model has an accuracy of 80% shows that it significantly outperforms this baseline, demonstrating its ability to provide more nuanced and accurate predictions.

**Why These Metrics Matter**

These metrics help you understand the strengths and limitations of the model. While the model is highly accurate and reliable, it’s important to remember that these predictions are based on historical data and patterns. Every case is unique, and factors that weren’t considered by the model could influence the actual outcome. Therefore, it’s essential to use the model’s predictions as a guide rather than a definitive answer.