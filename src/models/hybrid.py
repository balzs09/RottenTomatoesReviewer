import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdflib import Graph
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "rt_sample.csv"
DEFAULT_TTL_PATH = ROOT_DIR / "data" / "movies_knowledge_graph.ttl"

from src.evaluator.evaluator import evaluate_predictions

#Lokáis RDF tudásgráf betöltése és számszerű jellemzők kinyerése SPARQL-lel.
def load_features_from_ttl(ttl_path: Path) -> pd.DataFrame:
    print(f"Lokális RDF tudásgráf betöltése: {ttl_path}")
    g = Graph()
    g.parse(ttl_path, format="turtle")

    query = """
    PREFIX ex: <http://example.org/rotten_tomatoes/>
    SELECT ?rt_link ?dir_pos ?dir_neg ?film_pos ?film_neg ?has_neg 
           (GROUP_CONCAT(?genre; separator=",") AS ?genres)
    WHERE {
        ?movie ex:rtlink ?rt_link ;
               ex:directorAwardsPos ?dir_pos ;
               ex:directorAwardsNeg ?dir_neg ;
               ex:filmAwardsPos ?film_pos ;
               ex:filmAwardsNeg ?film_neg .
        OPTIONAL { ?movie ex:hasGenre ?genre. } 
    }
    GROUP BY ?rt_link ?dir_pos ?dir_neg ?film_pos ?film_neg ?has_neg
    """

    rows = []
    for row in g.query(query):
        rows.append({
            "rotten_tomatoes_link": str(row.rt_link),
            "director_awards_pos": int(row.dir_pos),
            "director_awards_neg": int(row.dir_neg),
            "film_awards_pos" : int(row.film_pos),
            "film_awards_neg": int(row.film_neg),
            "wikidata_genres": str(row.genres) if row.genres else "",
        })

    print(f"Gráfból kinyert egyedi filmek száma: {len(rows)}")
    return pd.DataFrame(rows)

def train_hybrid(data_path: Path = DEFAULT_DATA_PATH, ttl_path: Path = DEFAULT_TTL_PATH):
    print("Adathalmaz betöltése és összekapcsolása")
    df_rt = pd.read_csv(data_path)
    df_wiki = load_features_from_ttl(ttl_path)
    df =pd.merge(df_rt, df_wiki, on="rotten_tomatoes_link", how="inner")

    #Hiányzó értékek kezelése
    num_cols = [
        "director_awards_pos",
        "director_awards_neg",
        "film_awards_pos",
        "film_awards_neg",
    ]
    df[num_cols] = df[num_cols].fillna(0)
    df['wikidata_genres'] = df['wikidata_genres'].fillna("")
    df['review_content'] = df['review_content'].fillna("")
    print(f"Összekapcsolt minták száma: {len(df)}")

    # Index alapú szétválasztás
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    #3 Különálló feature-csoport egyesítése egyetlen mátrixba
    preprocessor = ColumnTransformer(
        transformers=[
            # 1. Szövehes jellemzők: a kritikák TF-IDF vektorizálása
            (
                "text",
                TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
                'review_content',
            ),
            #2. Számszerű Wikidata jellemzők: log1p a ferde eloszlás tompítására, majd standardizálás a TF-IDF skálájához
            (
                "wiki_num",
                Pipeline([
                    ("log",FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scaler",StandardScaler()),
                ]),
                num_cols
            ),
            #3. Műfajok: bináris kódolás; a regex egyben tartja a többszavas múfajokat
            (

                "wiki_genres",
                CountVectorizer(token_pattern=r"[^,]+", binary=True, min_df=500),
                "wikidata_genres",
            ),
        ],
        transformer_weights={
            "text": 1.0,
            "wiki_num": 1.0,
            "wiki_genres": 1.0,
        }

    )

    # Pipeline: előfeldolgozás és ugyanaz a LogisticRegression, mint az ml_baseline modellben lévő
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    C=4
                )
            )
        ]
    )

    print("Hibrid modell tanítása")
    pipeline.fit(df_train, df_train['label'])

    #Kiértékelés a teszthalmazon
    y_pred = pipeline.predict(df_test)
    y_proba = pipeline.predict_proba(df_test)[:, 1]
    evaluate_predictions(df_test, df_test['label'], y_pred, model_name="HYBRID MODEL", y_proba = y_proba)

    #Megtanult súlyok kiíratása
    clf = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = clf.coef_[0]

    print("\n Wikidata számszerű jellemzők megtanult súlyai")
    genre_weights = []
    for name, weight in zip(feature_names, coefs):
        if name.startswith("wiki_num__"):
            clean_name = name.replace("wiki_num__", "")
            direction = "Fresh" if weight > 0 else "Rotten"
            print(f"{clean_name:22s}: {weight:+.4f} -> {direction}")
        if name.startswith("wiki_genres__"):
            clean_genre_name = name.replace("wiki_genres__", "").strip()
            genre_weights.append((clean_genre_name, weight))

    genre_weights.sort(key=lambda x: x[1], reverse=True)
    print("\n Top 5 Fresh-be tartozó műfaj")
    for g, w in genre_weights[:5]:
        print(f"{g:30s}: {w:+.4f}")
    print("\n Bottom 5 Rotten-be tartozó műfaj")
    for g, w in  reversed(genre_weights[-5:]):
        print(f"{g:30s}: {w:+.4f}")

    return pipeline

def load_merged_dataset(data_path: Path = DEFAULT_DATA_PATH, ttl_path: Path = DEFAULT_TTL_PATH) -> pd.DataFrame:
    df_rt = pd.read_csv(data_path)
    df_wiki = load_features_from_ttl(ttl_path)
    df = pd.merge(df_rt, df_wiki, on="rotten_tomatoes_link", how="inner")

    num_cols = [
        "director_awards_pos",
        "director_awards_neg",
        "film_awards_pos",
        "film_awards_neg",
    ]
    df[num_cols] = df[num_cols].fillna(0)
    df["wikidata_genres"] = df["wikidata_genres"].fillna("")
    df["review_content"] = df["review_content"].fillna("")
    return df


def evaluate_group_kfold(df: pd.DataFrame, n_splits: int = 5, C: float = 4.0):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["rotten_tomatoes_link"]
    num_cols = [
        "director_awards_pos",
        "director_awards_neg",
        "film_awards_pos",
        "film_awards_neg",
    ]

    configs = {
        "Csak szöveg (wiki=0)": {"text": 1.0, "wiki_num": 0.0, "wiki_genres": 0.0},
        "Hibrid (szöveg + Wikidata)": None,
    }

    for name, weights in configs.items():
        accs, maes = [], []

        for train_idx, test_idx in gkf.split(df, df["label"], groups=groups):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "text",
                        TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2)),
                        "review_content",
                    ),
                    (
                        "wiki_num",
                        Pipeline([
                            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("scaler", StandardScaler()),
                        ]),
                        num_cols,
                    ),
                    (
                        "wiki_genres",
                        CountVectorizer(token_pattern=r"[^,]+", binary=True, min_df=500),
                        "wikidata_genres",
                    ),
                ],
                transformer_weights=weights,
            )

            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42, C=C)),
            ])

            pipe.fit(df_train, df_train["label"])
            y_pred = pipe.predict(df_test)
            y_proba = pipe.predict_proba(df_test)[:, 1]

            accs.append(accuracy_score(df_test["label"], y_pred) * 100)

            df_test["pred_proba"] = y_proba
            film_stats = df_test.groupby("rotten_tomatoes_link").agg(
                real=("label", lambda s: s.mean() * 100),
                pred=("pred_proba", lambda s: s.mean() * 100),
            )
            maes.append(np.mean(np.abs(film_stats["real"] - film_stats["pred"])))

        print(f"\n--- {name} ({n_splits}-fold GroupKFold, C={C}) ---")
        print(f"Pontosság : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"Film MAE  : {np.mean(maes):.2f}% ± {np.std(maes):.2f}%")
        print(f"Fold MAE-k: {[round(m, 2) for m in maes]}")

if __name__ == "__main__":
    df = load_merged_dataset()
    #evaluate_group_kfold(df)
    train_hybrid()


