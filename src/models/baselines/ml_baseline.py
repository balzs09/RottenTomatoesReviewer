import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "rt_sample.csv"

from src.evaluator.evaluator import evaluate_predictions

def train_baseline(data_path: Path = DEFAULT_DATA_PATH):
    print("Minta adathalmaz betöltése.")
    df = pd.read_csv(data_path)

    #X bemenet és y célváltozó
    X = df['review_content']
    y = df['label']

    #Tanító és teszt halmaz szétválasztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Tanító minták: {len(X_train)}, Teszt minták: {len(X_test)}")

    #Szövegek számmá alakízása TF-IDF-fel.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
    )
    print("TF-IDF vektorizáció folyamatban.")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Modell tanítása")
    #A saját class weight által nagyobb súlyt kapnak a ritka "Rotten" kritikák a filmek pozítív eltolódásának megoldásaként.
    model = LogisticRegression(max_iter=1000, random_state=42,class_weight={0: 2.0, 1: 1.0})
    model.fit(X_train_vec, y_train)

    #Adatfelosztás indexek alpján
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
    df_test = df.loc[test_idx]

    # Kiértékelés a teszthalmazon
    y_pred = model.predict(X_test_vec)
    evaluate_predictions(df_test, df_test['label'], y_pred, model_name="ML BASELINE")

    return model,vectorizer

if __name__ == "__main__":
    model, vectorizer = train_baseline()