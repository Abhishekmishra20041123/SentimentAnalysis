# Twitter Sentiment Analysis - 4 classes
# Run: python train_model.py

import os
import re
import string
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from joblib import Parallel, delayed
import nltk
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)
CACHE_FILE = MODEL_DIR / 'preprocessed_data_twitter.joblib'

VALID_LABELS = {'positive', 'negative', 'neutral', 'irrelevant'}
LABEL_MAP = {
    'positive': 'positive', 'Positive': 'positive',
    'negative': 'negative', 'Negative': 'negative',
    'neutral': 'neutral', 'Neutral': 'neutral',
    'irrelevant': 'irrelevant', 'Irrelevant': 'irrelevant',
}


def gpu_available():
    try:
        import xgboost as xgb
        booster = xgb.Booster({'device': 'cuda'})
        del booster
        return True
    except Exception:
        return False


USE_GPU = gpu_available()


def get_xgb_classifier(num_classes):
    params = dict(
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0,
        objective='multi:softprob',
        num_class=num_classes,
        tree_method='hist',
    )
    if USE_GPU:
        params['device'] = 'cuda'
    return XGBClassifier(**params)


def load_twitter_data():
    """Load Twitter dataset (pos/neg/neutral/irrelevant)."""
    twitter = pd.read_csv(
        BASE_DIR / 'twitter_sentiment.csv',
        header=None,
        names=['tweet_id', 'context', 'sentiment', 'text'],
    )
    twitter['source'] = 'twitter'
    twitter['sentiment'] = twitter['sentiment'].map(LABEL_MAP)

    twitter = twitter.dropna(subset=['text', 'sentiment'])
    twitter['text'] = twitter['text'].astype(str).str.strip()
    twitter = twitter[twitter['sentiment'].isin(VALID_LABELS)]
    twitter = twitter[twitter['text'].str.split().str.len() >= 2]
    twitter = twitter.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    return twitter


def balance_dataset(df, random_state=42):
    """
    Downsample positive & negative to match the largest minority class (neutral/irrelevant).
    Keeps all neutral and irrelevant samples; reduces pos/neg to match minority class size.
    """
    print('\nBefore balancing:')
    print(df['sentiment'].value_counts())

    minority_target = int(
        df[df['sentiment'].isin(['neutral', 'irrelevant'])]['sentiment'].value_counts().max()
    )
    parts = []
    for label in sorted(VALID_LABELS):
        subset = df[df['sentiment'] == label]
        if label in ('positive', 'negative') and len(subset) > minority_target:
            subset = subset.sample(n=minority_target, random_state=random_state)
        parts.append(subset)

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f'\nAfter balancing (pos/neg capped at {minority_target:,} each):')
    print(balanced['sentiment'].value_counts())
    print('Total samples:', len(balanced))
    return balanced


def clean_text_series(series):
    s = series.str.lower()
    s = s.str.replace(r'<br\s*/?>', ' ', regex=True)
    s = s.str.replace(r'<[^>]+>', ' ', regex=True)
    s = s.str.replace(r'http\S+|www\S+', ' ', regex=True)
    s = s.str.replace(r'@\w+', ' ', regex=True)
    s = s.str.replace(r'\d+', ' ', regex=True)
    s = s.str.translate(str.maketrans('', '', string.punctuation))
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def preprocess_tokens(review, stop_words_set, lemmatizer_obj):
    tokens = word_tokenize(review)
    tokens = [lemmatizer_obj.lemmatize(t) for t in tokens if t not in stop_words_set and len(t) > 1]
    return ' '.join(tokens)


def multiclass_scores(y_true, y_pred, y_proba):
    return {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'),
    }


def make_vectorizer(max_features):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_feature_caches(text_splits, max_features_list):
    """Precompute TF-IDF matrices once per max_features (major speedup vs Pipeline CV)."""
    caches = {}
    for max_feat in max_features_list:
        vec = make_vectorizer(max_feat)
        caches[max_feat] = {
            'vectorizer': vec,
            'train': vec.fit_transform(text_splits['train']),
            'tune': vec.transform(text_splits['tune']),
            'test': vec.transform(text_splits['test']),
        }
    return caches


def rebuild_pipeline(vectorizer, classifier, svd=None):
    steps = [
        ('column_transformer', ColumnTransformer([
            ('tfidf', vectorizer, 'review_clean'),
        ], remainder='drop')),
    ]
    if svd is not None:
        steps.append(('svd', svd))
    steps.append(('classifier', classifier))
    return Pipeline(steps)


print('Working directory:', BASE_DIR)
print('GPU available:', USE_GPU, '(XGBoost will use CUDA)' if USE_GPU else '(XGBoost on CPU)')


def main():
    if CACHE_FILE.exists():
        print('Loading cached preprocessed data...')
        cached = joblib.load(CACHE_FILE)
        data = cached['data']
        stop_words = cached['stop_words']
        lemmatizer = cached['lemmatizer']
        print('Cache loaded! Samples:', len(data))
    else:
        data = load_twitter_data()
        data = balance_dataset(data)
        print('Twitter dataset shape:', data.shape)

        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        data['review_clean'] = clean_text_series(data['text'])
        data = data[data['review_clean'].str.len() > 0]

        print('Preprocessing text (parallel)...')
        cleaned_reviews = Parallel(n_jobs=-1, prefer='threads')(
            delayed(preprocess_tokens)(review, stop_words, lemmatizer)
            for review in tqdm(data['review_clean'], desc='NLTK cleaning')
        )
        data['review_clean'] = cleaned_reviews
        data = data[data['review_clean'].str.len() > 0]
        joblib.dump({'data': data, 'stop_words': stop_words, 'lemmatizer': lemmatizer}, CACHE_FILE)
        print('Preprocessing cached to', CACHE_FILE)

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(VALID_LABELS))
    X = data[['review_clean']]
    Y = label_encoder.transform(data['sentiment'])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y,
    )
    tune_size = min(20000, len(X_train))
    X_tune, _, Y_tune, _ = train_test_split(
        X_train, Y_train, train_size=tune_size, random_state=42, stratify=Y_train,
    )
    print('Train:', X_train.shape, 'Test:', X_test.shape, 'Tune:', X_tune.shape)
    print('Classes:', list(label_encoder.classes_))

    num_classes = len(label_encoder.classes_)
    max_features_list = [5000, 8000]

    print('\nPrecomputing TF-IDF features (avoids repeated vectorization)...')
    feat_cache = build_feature_caches(
        {
            'train': X_train['review_clean'],
            'tune': X_tune['review_clean'],
            'test': X_test['review_clean'],
        },
        max_features_list,
    )

    xgb_n_jobs = 1 if USE_GPU else -1

    # Logistic Regression
    print('\n--- Logistic Regression ---')
    lr_best = None
    lr_best_score = -1
    lr_best_mf = 8000
    for mf in max_features_list:
        X_tune_vec = feat_cache[mf]['tune']
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        search = RandomizedSearchCV(lr, {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
        }, n_iter=4, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
           scoring='accuracy', n_jobs=-1, random_state=42)
        search.fit(X_tune_vec, Y_tune)
        if search.best_score_ > lr_best_score:
            lr_best_score = search.best_score_
            lr_best = search.best_estimator_
            lr_best_mf = mf
            lr_search = search
    print('Best CV Accuracy:', lr_best_score, '| max_features:', lr_best_mf)

    # Random Forest
    print('\n--- Random Forest ---')
    rf_best = None
    rf_best_score = -1
    rf_best_mf = 8000
    for mf in max_features_list:
        X_tune_vec = feat_cache[mf]['tune']
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        search = RandomizedSearchCV(rf, {
            'n_estimators': [100, 200],
            'max_depth': [None, 30],
            'min_samples_split': [2, 5],
        }, n_iter=4, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
           scoring='accuracy', n_jobs=-1, random_state=42)
        search.fit(X_tune_vec, Y_tune)
        if search.best_score_ > rf_best_score:
            rf_best_score = search.best_score_
            rf_best = search.best_estimator_
            rf_best_mf = mf
            rf_search = search
    print('Best CV Accuracy:', rf_best_score, '| max_features:', rf_best_mf)

    # HistGradientBoosting (much faster than sklearn GradientBoosting)
    print('\n--- Hist Gradient Boosting ---')
    gb_best = None
    gb_best_score = -1
    gb_best_mf = 5000
    gb_best_svd = 200
    for mf in max_features_list:
        X_tune_vec = feat_cache[mf]['tune']
        svd = TruncatedSVD(n_components=200, random_state=42)
        X_tune_svd = svd.fit_transform(X_tune_vec)
        hgb = HistGradientBoostingClassifier(random_state=42)
        search = RandomizedSearchCV(hgb, {
            'max_iter': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }, n_iter=4, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
           scoring='accuracy', n_jobs=-1, random_state=42)
        search.fit(X_tune_svd, Y_tune)
        if search.best_score_ > gb_best_score:
            gb_best_score = search.best_score_
            gb_best = search.best_estimator_
            gb_best_mf = mf
            gb_best_svd = svd
            gb_search = search
    print('Best CV Accuracy:', gb_best_score, '| max_features:', gb_best_mf)

    # XGBoost (GPU when available)
    print('\n--- XGBoost', '(GPU)' if USE_GPU else '(CPU)', '---')
    xgb_best = None
    xgb_best_score = -1
    xgb_best_mf = 8000
    for mf in max_features_list:
        X_tune_vec = feat_cache[mf]['tune']
        xgb = get_xgb_classifier(num_classes)
        search = RandomizedSearchCV(xgb, {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
        }, n_iter=4, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
           scoring='accuracy', n_jobs=xgb_n_jobs, random_state=42)
        search.fit(X_tune_vec, Y_tune)
        if search.best_score_ > xgb_best_score:
            xgb_best_score = search.best_score_
            xgb_best = search.best_estimator_
            xgb_best_mf = mf
            xgb_search = search
    print('Best CV Accuracy:', xgb_best_score, '| max_features:', xgb_best_mf)

    print('\nTraining on full training data...')
    vec_lr = make_vectorizer(lr_best_mf)
    X_train_lr = vec_lr.fit_transform(X_train['review_clean'])
    X_test_lr = vec_lr.transform(X_test['review_clean'])
    lr_best.fit(X_train_lr, Y_train)

    vec_rf = make_vectorizer(rf_best_mf)
    X_train_rf = vec_rf.fit_transform(X_train['review_clean'])
    X_test_rf = vec_rf.transform(X_test['review_clean'])
    rf_best.fit(X_train_rf, Y_train)

    vec_gb = make_vectorizer(gb_best_mf)
    X_train_gb = vec_gb.fit_transform(X_train['review_clean'])
    X_test_gb = vec_gb.transform(X_test['review_clean'])
    gb_svd = TruncatedSVD(n_components=gb_best_svd.n_components, random_state=42)
    X_train_gb_svd = gb_svd.fit_transform(X_train_gb)
    X_test_gb_svd = gb_svd.transform(X_test_gb)
    gb_best.fit(X_train_gb_svd, Y_train)

    vec_xgb = make_vectorizer(xgb_best_mf)
    X_train_xgb = vec_xgb.fit_transform(X_train['review_clean'])
    X_test_xgb = vec_xgb.transform(X_test['review_clean'])
    xgb_best.fit(X_train_xgb, Y_train)

    fitted_models = {
        'Logistic Regression': (lr_best, vec_lr, None, X_train_lr, X_test_lr),
        'Random Forest': (rf_best, vec_rf, None, X_train_rf, X_test_rf),
        'Hist Gradient Boosting': (gb_best, vec_gb, gb_svd, X_train_gb_svd, X_test_gb_svd),
        'XGBoost': (xgb_best, vec_xgb, None, X_train_xgb, X_test_xgb),
    }

    results = []
    pipelines = {}
    for name, (clf, vec, svd, X_tr, X_te) in fitted_models.items():
        train_pred = clf.predict(X_tr)
        test_pred = clf.predict(X_te)
        test_proba = clf.predict_proba(X_te)
        scores = multiclass_scores(Y_test, test_pred, test_proba)
        results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(Y_train, train_pred),
            'Test Accuracy': accuracy_score(Y_test, test_pred),
            'Precision': scores['precision'],
            'Recall': scores['recall'],
            'F1 Score': scores['f1'],
            'ROC AUC': scores['roc_auc'],
        })
        pipelines[name] = rebuild_pipeline(vec, clf, svd)

    # Refit pipelines on raw text for Streamlit joblib compatibility
    for name, pipe in pipelines.items():
        pipe.fit(X_train, Y_train)

    results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
    print('\n', results_df)

    best_model_name = results_df.iloc[0]['Model']
    best_model = pipelines[best_model_name]
    best_test_accuracy = results_df.iloc[0]['Test Accuracy']
    print('\nBest Model:', best_model_name, '| Test Accuracy:', best_test_accuracy)

    dataset_stats = {
        'total_samples': int(len(data)),
        'twitter_samples': int(len(data)),
        'class_distribution': data['sentiment'].value_counts().to_dict(),
        'balanced': True,
        'balancing': 'positive/negative downsampled to match largest minority class',
        'classes': list(label_encoder.classes_),
        'num_classes': num_classes,
        'gpu_used': USE_GPU,
    }

    # Save all models separately for model selection in UI
    for name, pipe in pipelines.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(pipe, MODEL_DIR / f'model_{safe_name}.joblib')
        print(f'Saved {name} to models/model_{safe_name}.joblib')
    
    # Save best model with original name for backward compatibility
    joblib.dump(best_model, MODEL_DIR / 'sentiment_model.joblib')
    joblib.dump(label_encoder, MODEL_DIR / 'label_encoder.joblib')
    joblib.dump(stop_words, MODEL_DIR / 'stop_words.pkl')
    joblib.dump(lemmatizer, MODEL_DIR / 'lemmatizer.pkl')
    joblib.dump(results_df, MODEL_DIR / 'model_results.pkl')

    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'best_test_accuracy': float(best_test_accuracy),
            'num_classes': num_classes,
            'classes': list(label_encoder.classes_),
            'dataset': dataset_stats,
            'gpu_used': USE_GPU,
            'all_models': results_df.to_dict(orient='records'),
            'tuning': {
                'Logistic Regression': {'best_cv_accuracy': float(lr_best_score), 'best_params': lr_search.best_params_},
                'Random Forest': {'best_cv_accuracy': float(rf_best_score), 'best_params': rf_search.best_params_},
                'Hist Gradient Boosting': {'best_cv_accuracy': float(gb_best_score), 'best_params': gb_search.best_params_},
                'XGBoost': {'best_cv_accuracy': float(xgb_best_score), 'best_params': xgb_search.best_params_},
            },
            'dataset_source': 'Twitter only',
        }, f, indent=2)

    print('\nModel saved to models/sentiment_model.joblib')
    print('Classes:', list(label_encoder.classes_))
    print('Now run UI: streamlit run app.py')


if __name__ == '__main__':
    main()
