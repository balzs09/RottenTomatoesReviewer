import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "rt_sample.csv"

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
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    #Kiértékelés a teszthalmazon
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- ML BASELINE EREDMÉNYEK ---")
    print(f"Pontosság: {accuracy*100:.2f}%\n")
    print("Részletes metrikák:")
    print(classification_report(y_test, y_pred, target_names=['Rotten', 'Fresh']))
    print("Konfúziós mátrix:")
    print(confusion_matrix(y_test, y_pred))

    return model,vectorizer

if __name__ == "__main__":
    model, vectorizer = train_baseline()