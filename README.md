# Customer-Segmentation-Sentiment-Analysis-Project
This project performs text-based sentiment analysis and exploratory product insights using e-commerce product reviews.
The goal is to understand customer opinions and evaluate product performance based on real customer feedback.
-----------------------------------------------------
Key Objectives

* Clean & preprocess product review data
* Perform exploratory data analysis (EDA)
* Visualize sentiment patterns & top words
* Convert text into numerical features (TF-IDF)
* Handle class imbalance with SMOTE
* Train multiple ML models
* Tune hyperparameters (XGBoost)
---------------------------------------------------------
📂 Dataset Description
| Column       | Description                           |
| ------------ | ------------------------------------- |
| ProductName  | Name of product                       |
| ProductPrice | Listed price                          |
| Rate         | User rating (1–5)                     |
| Review       | Customer review text                  |
| Summary      | Short summary                         |
| Sentiment    | Label (positive / negative / neutral) |
Initial rows: 171,380
Final usable rows after cleaning: 144,871
------------------------------------------
🧹 Data Cleaning

Steps performed:

✔ Convert price & rating to numeric
✔ Remove special characters & lowercase text
✔ Drop duplicates
✔ Handle missing values
✔ Normalize sentiment labels (Positive/Neutral/Negative)
✔ Combine columns into one text field: Review + Summary
-----------------------------------------------
📊 Exploratory Data Analysis

Visualizations performed:

Sentiment distribution

Product price distribution

Ratings by sentiment

Word clouds for each sentiment

Most common words & bigrams

Correlation map (price vs rating)

Review length analysis

Key Insights:

~61% reviews are positive

Positive reviews correlate with higher star ratings (avg ~4.6)

Negative reviews use words like bad, worst, waste, not

Popular bigrams: good product, nice product, waste money
---------------------------------------------------
🧠 Machine Learning Models
Features & Sampling

Vectorizer: TF-IDF (5000 features)

Imbalance handling: SMOTE

Train/Test split: 80/20

Models Trained
| Model                                | Accuracy   |
| ------------------------------------ | ---------- |
| Logistic Regression                  | 0.8340     |
| Random Forest                        | **0.9238** |
| Naive Bayes                          | 0.8115     |
| XGBoost                              | 0.8440     |
| ✅ Tuned XGBoost (best general model) | **0.8763** |
----------------------------------------------------
Best Model

Tuned XGBoost Classifier

Macro F1 ~0.88

Strong performance across all sentiment classes

Identified key words influencing sentiment

Top Keywords Detected

Positive: amazing, excellent, love, awesome
Negative: worst, waste, horrible, useless, bad
-----------------------------------------------------
💾 Model Export

Saved artifacts using joblib:

sentiment_xgb_model.pkl

tfidf_vectorizer.pkl

👀 Sample prediction included in notebook.
-----------------------------------------------------------
📎 Tech Stack
| Category        | Tools                          |
| --------------- | ------------------------------ |
| Data            | Pandas, NumPy                  |
| NLP             | NLTK, Regex, WordCloud, TF-IDF |
| ML              | Scikit-Learn, XGBoost, SMOTE   |
| Visualization   | Matplotlib, Seaborn            |
| Deployment Prep | joblib                         |

├── data/
├── sentiment_project.ipynb
├── sentiment_xgb_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── requirements.txt

-------------------------------------------------------------
✅ Achievements

Built complete NLP sentiment pipeline

Performed deep EDA + word analytics

Achieved >87% accuracy

Exported model & vectorizer

Demonstrated ML experimentation & tuning

🚀 Future Work

Add real customer-level segmentation (RFM clustering)

Build a Streamlit or Flask UI

Deploy as API for live sentiment prediction

Improve text preprocessing (negation handling, lemmatization)

Try transformer models (BERT / DistilBERT)

🙌 Acknowledgements

Kaggle / public e-commerce review dataset

Scikit-learn, XGBoost, NLTK


Evaluate model performance

Export trained model for deployment
