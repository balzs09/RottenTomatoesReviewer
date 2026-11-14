import pandas as pd
import sys
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "rt_sample.csv"

from src.evaluator.evaluator import evaluate_predictions


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
    if positive_count >= negative_count:
        return 1
    else:
        return 0

#Kritikák kiértékelése érzelembecslés alapú szótár alapján
def run_lexicon_baseline(data_path: Path = DEFAULT_DATA_PATH) :
    print("Minta adathalmaz betöltése...")
    df = pd.read_csv(data_path)

    #Adatfelosztás indexek alpján
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
    df_test = df.loc[test_idx]

    # Kiértékelés a teszthalmazon
    y_true = df_test['label']
    y_pred = df_test['review_content'].apply(predict_sentiment_lexicon)
    evaluate_predictions(df_test, y_true, y_pred, model_name="LEXICON_BASELINE")

if __name__ == "__main__":
    run_lexicon_baseline()
