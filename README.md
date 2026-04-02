# Credit Card Fraud Detection

## About the Project
This project focuses on detecting fraudulent credit card transactions using machine learning.  
The dataset is highly imbalanced, so the main focus was on proper evaluation and improving real-world performance rather than just accuracy.


## Goals
- Build a model to detect fraud  
- Handle class imbalance correctly  
- Understand evaluation metrics  
- Fine-tune thresholds


## Tools
- Python  
- scikit-learn  
- XGBoost  
- Pandas, NumPy  


## Approach
I trained LogisticRegression, RaandomForest and XGBoost classifier on diffrent number of features and experimented with hyperparameters to get better results. 

Instead of relying on accuracy, I evaluated the model using:
- Precision  
- Recall  
- F1-score  
- PR-AUC  

I also adjusted the prediction threshold using predicted probabilities:

```python
prob = model.predict_proba(X)[:, 1]
preds = (prob >= threshold).astype(int)
```

## Results
- Precision: ~0.88–0.98  
- Recall: ~0.76–0.79  
- F1-score: ~0.83–0.86  
- PR-AUC: ~0.82–0.84  


## Key Takeaways
- Accuracy is not reliable for imbalanced datasets
- Threshold tuning can significantly impact performance  
- XGBoost requires regularization to avoid overfitting 