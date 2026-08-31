# Sentiment Analysis Dashboard

A comprehensive Twitter sentiment analysis dashboard using machine learning with TF-IDF features. The system classifies tweets into 4 sentiment categories: Positive, Negative, Neutral, and Irrelevant.

## Features

- **4-Class Classification**: Positive, Negative, Neutral, Irrelevant sentiment detection
- **Multiple ML Models**: Logistic Regression, Random Forest, Hist Gradient Boosting, XGBoost
- **Interactive Dashboard**: Streamlit-based UI with data visualization
- **Real-time Prediction**: Live sentiment prediction with confidence scores
- **Model Comparison**: Detailed performance metrics and visualizations
- **Data Analysis**: Word clouds, distribution plots, and statistical analysis

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment_Analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install the key packages manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn nltk matplotlib plotly wordcloud joblib xgboost
   ```

## Setup

1. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
   ```

2. **Prepare the dataset**
   - Place your Twitter sentiment dataset as `twitter_sentiment.csv` in the project root
   - The CSV should have columns: `tweet_id`, `context`, `sentiment`, `text`

3. **Train the model**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Train 4 different ML models
   - Perform hyperparameter tuning
   - Save models and artifacts to the `models/` directory
   - Generate evaluation metrics

## Running the Application

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

The application will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501

## Dashboard Pages

### 1. Overview
- Dataset statistics and sentiment distribution
- Key metrics and sample reviews
- Quick data summary table

### 2. Data Analysis
- Word count and review length distributions
- Comparative analysis across sentiment classes
- Word clouds for each sentiment category
- Top frequent words analysis

### 3. Model Evaluation
- Performance comparison across all models
- Train vs Test accuracy charts
- Precision, Recall, F1 Score, ROC AUC metrics
- Radar charts and heatmaps
- Hyperparameter tuning results

### 4. Prediction
- Live sentiment prediction on custom text
- Model selection dropdown
- Confidence scores and probability distributions
- Pre-built example tweets for testing

### 5. About
- Project information and technical details
- Model performance summary
- Sentiment class descriptions

## Project Structure

```
Sentiment_Analysis/
├── app.py                 # Streamlit dashboard application
├── train_model.py         # Model training script
├── test_notebook.py       # Testing/evaluation notebook
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── twitter_sentiment.csv # Dataset file
└── models/               # Saved model artifacts
    ├── sentiment_model.joblib
    ├── model_random_forest.joblib
    ├── model_logistic_regression.joblib
    ├── model_hist_gradient_boosting.joblib
    ├── model_xgboost.joblib
    ├── label_encoder.joblib
    ├── stop_words.pkl
    ├── lemmatizer.pkl
    ├── model_results.pkl
    └── metrics.json
```

## Model Performance

Current best model: **Random Forest** (88.67% test accuracy)

| Model | Test Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|---------------|-----------|--------|----------|---------|
| Random Forest | 88.67% | 88.68% | 88.67% | 88.67% | 97.78% |
| Logistic Regression | 76.22% | 76.28% | 76.22% | 76.23% | 92.89% |
| Hist Gradient Boosting | 65.69% | 65.71% | 65.69% | 65.42% | 87.31% |
| XGBoost | 61.47% | 62.32% | 61.47% | 60.91% | 84.74% |

## Usage Tips

- **Model Selection**: Use Logistic Regression for better neutral/irrelevant detection, Random Forest for higher overall accuracy
- **Preprocessing**: The model uses NLTK lemmatization and stopword removal
- **Features**: TF-IDF vectorization with character and word n-grams
- **Balanced Dataset**: Positive/negative classes are downsampled to match minority classes

## Troubleshooting

**Model not found error:**
```bash
# Run the training script first
python train_model.py
```

**NLTK data missing:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

**Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning models and preprocessing
- **NLTK**: Natural language processing and tokenization
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive data visualization
- **Matplotlib**: Static plotting and word clouds
- **Pandas**: Data manipulation and analysis

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
