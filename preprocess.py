import pandas as pd

def load_and_sample_data(movies_path: str, reviews_path: str, sample_size: int = 100):
    # Releváns oszlopok beolvasása
    print("CSV fájlok beolvasása...")
    movies_df = pd.read_csv(movies_path,usecols=['rotten_tomatoes_link', 'movie_title'])
    reviews_df = pd.read_csv(reviews_path,usecols=['rotten_tomatoes_link','review_content','review_type'])

    # Hiányos filmek és értékelések kiszűrése
    movies_df = movies_df.dropna(subset=['rotten_tomatoes_link','movie_title'])
    reviews_df = reviews_df.dropna(subset=['rotten_tomatoes_link','review_content','review_type'])
    reviews_df = reviews_df[reviews_df["review_type"].isin(['Fresh','Rotten'])]

    # A legtöbb kritikával rendelkező filmekből mintavételezés
    top_movie_links = movies_df['rotten_tomatoes_link'].value_counts().head(sample_size).index
    filtered_movies = movies_df[movies_df['rotten_tomatoes_link'].isin(top_movie_links)]
    filtered_reviews = reviews_df[reviews_df['rotten_tomatoes_link'].isin(top_movie_links)]

    # Adatok összefűzése
    dataset = pd.merge(filtered_movies, filtered_reviews, on='rotten_tomatoes_link')
    dataset['label'] = (dataset['review_type']=='Fresh').astype(int)

    # Eredmény visszatérítése
    output_filename = "rt_sample.csv"
    dataset.to_csv(output_filename, index=False)
    print(f"Mentve: {output_filename}. {len(dataset)} kritika, {dataset['movie_title'].nunique()}  film.")

if __name__ == '__main__':
    load_and_sample_data(
        movies_path="rotten_tomatoes_movies.csv",
        reviews_path="rotten_tomatoes_critic_reviews.csv",
        sample_size=150
    )