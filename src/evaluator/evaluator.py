import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

#Kiértékeli a modellt kritika szinten és film szinten aggregálva is.
def evaluate_predictions(df_eval: pd.DataFrame, y_true, y_pred, model_name: str = "Modell"):
    #Kritika szintű erdmények
    accuracy = accuracy_score(y_true, y_pred)
    print("\n---  {model_name.upper()} EREDMÉNYEK  ---")
    print(f"Pontosság: {accuracy * 100:.2f}%\n")
    print("Részletes metrikák:")
    print(classification_report(y_true, y_pred, target_names=['Rotten (0)', 'Fresh (1)']))
    print("Konfúziós mátrix:")
    print(confusion_matrix(y_true, y_pred))

    #Film szintű eredmények
    temp_df = df_eval.copy()
    temp_df['pred_label'] = y_pred

    #Prediktált fresh kritikák arányának kiszámítása
    movie_agg = temp_df.groupby('rotten_tomatoes_link').agg(
        movie_title=('movie_title', 'first'),
        real_tomatometer = ('tomatometer_rating', 'first'),
        pred_ratio =('pred_label', 'mean'),
        review_count = ('pred_label', 'count')
    ).dropna(subset=['real_tomatometer'])
    movie_agg['pred_tomatometer']= movie_agg['pred_ratio'] * 100

    # MAE (Mean Absolute Error) kiszámítása
    mae = mean_absolute_error(movie_agg['real_tomatometer'], movie_agg['pred_tomatometer'])
    movie_agg['diff']= abs(movie_agg['real_tomatometer'] - movie_agg['pred_tomatometer'])

    print(f"\n--- AGGREGÁLT TOMATOMETER KIÉRTÉKELÉS  ---")
    print(f"Tomatometer MAE (Átlagos abszolút hiba): {mae:.2f} %\n")
    print("Filmek valós vs. becsült pontszámai (Minta 15 filmből):")
    cols_to_show = ['movie_title', 'real_tomatometer', 'pred_tomatometer', 'diff', 'review_count']
    print(movie_agg[cols_to_show].head(15).to_string(index=False))