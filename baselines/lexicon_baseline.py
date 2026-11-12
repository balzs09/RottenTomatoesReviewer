import pandas as pd
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path
POSITIVE_WORDS = {
    'good', 'great', 'awesome', 'amazing', 'excellent', 'fantastic', 'masterpiece',
    'brilliant', 'fresh', 'love', 'best', 'enjoyable', 'compelling', 'entertaining',
    'charming', 'clever', 'funny', 'powerful', 'superb', 'wonderfully','exceptional'
}

NEGATIVE_WORDS = {
    'bad', 'awful', 'terrible', 'horrible', 'worst', 'boring', 'dull', 'waste',
    'rotten', 'hate', 'poor', 'stupid', 'mess', 'flat', 'cliché', 'disappointing',
    'annoying', 'cheap', 'fails', 'weak', 'disgusting', 'amateur'
}

# A függvény kiszámolja a pozitív és negatív szavak számát a kritikákban.
# Ha több a pozitív mint a negatív, akkor Fresh, egyébként Rotten.
def predict_sentiment_lexicon(text: str) -> int:
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    positive_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    if positive_count > negative_count:
        return 1
    else:
        return 0

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "rt_sample.csv"
#Kritikák kiértékelése érzelembecslés alapú szótár alapján
def run_lexicon_baseline(data_path: Path = DEFAULT_DATA_PATH) :
    print("Minta adathalmaz betöltése...")
    df = pd.read_csv(data_path)

    y_true = df['label']
    y_pred = df['review_content'].apply(predict_sentiment_lexicon)
    acc = accuracy_score(y_true, y_pred)
    print("\n--- LEXIKON-ALAPÚ BASELINE EREDMÉNYEK  ---")
    print(f"Pontosság: {acc * 100:.2f}%\n")
    print("Részletes metrikák:")
    print(classification_report(y_true, y_pred, target_names=['Rotten (0)', 'Fresh (1)']))
    print("Konfúziós mátrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    run_lexicon_baseline()
