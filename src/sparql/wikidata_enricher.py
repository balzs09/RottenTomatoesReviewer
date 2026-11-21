import sys
import time
from pathlib import Path
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "RottenTomatoesHybridML/1.0 (BME VIK Project; nagyadambalazs@edu.bme.hu)"



#Entity linking és metaadat kinyerés a Wikidata-ról SPARQLWrapperrrel,
def fetch_movie_data(movie_title: str, target_year: int) :
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT,agent=USER_AGENT)
    sparql.setReturnFormat(JSON)

    #Viágpremierek és fesztiválok miatt meg van engedve +-1 év eltérés a paraméterként megadott megjelnési évhez képest.
    year_filter = f"FILTER(ABS(?releaseYear - {target_year}) <=1)"
    safe_title = movie_title.replace('"', '\\"')
    print(f"{movie_title} ({target_year}) keresése a Wikidatan.")

    #SPARQL lekérdezés egy filmre a film címe és opcionálisan a megjelenési év alapján
    sparql_query = f"""
    SELECT ?movie
           (SAMPLE(?director) AS ?dir)
           (SAMPLE(?directorLabel) AS ?dirName)
           (Count(DISTINCT ?posDirAward) AS ?directorAwardsPos)
           (Count(DISTINCT ?negDirAward) AS ?directorAwardsNeg)
           (Count(DISTINCT ?posMovieAward) AS ?filmAwardsPos)
           (Count(DISTINCT ?negMovieAward) AS ?filmAwardsNeg)
           (GROUP_CONCAT(DISTINCT ?genreLabel; SEPARATOR=", ") AS ?genres)
    WHERE {{
      {{
        SELECT ?movie ?sitelinks WHERE {{
          VALUES ?title {{"{safe_title}"@en "{safe_title}"@mul}}
          ?movie rdfs:label ?title.
          ?movie wdt:P577 ?date;
                 wdt:P57 [] ;
                 wikibase:sitelinks ?sitelinks.
          BIND(YEAR(?date) AS ?releaseYear)
          {year_filter}
          FILTER (
            EXISTS {{ ?movie wdt:P31 wd:Q11424}} ||
            EXISTS {{ ?movie wdt:P31 ?type . ?type wdt:P279+ wd:Q11424 }} 
          )
          }}
          GROUP BY ?movie ?sitelinks
          ORDER BY DESC(?sitelinks)
          LIMIT 1
      }}
       OPTIONAL {{ 
        ?movie wdt:P166 ?posMovieAward. 
        ?posMovieAward rdfs:label ?posMovieAwardLabel .
        FILTER(LANG(?posMovieAwardLabel) = "en" && !CONTAINS(LCASE(?posMovieAwardLabel), "raspberry"))
      }}
      OPTIONAL {{
        ?movie wdt:P166 ?negMovieAward. 
        ?negMovieAward rdfs:label ?negMovieAwardLabel .
        FILTER(LANG(?negMovieAwardLabel) = "en" && CONTAINS(LCASE(?negMovieAwardLabel), "raspberry"))
        
      }}
      OPTIONAL {{
        ?movie wdt:P57 ?director.
        ?director rdfs:label ?directorLabel. FILTER(LANG(?directorLabel) IN ("en","mul")) 
        OPTIONAL {{ 
          ?director wdt:P166 ?posDirAward.
          ?posDirAward rdfs:label ?posDirAwardLabel.
          FILTER(LANG(?posDirAwardLabel) = "en" && !CONTAINS(LCASE(?posDirAwardLabel), "raspberry"))
        }}
        OPTIONAL {{ 
          ?director wdt:P166 ?negDirAward . 
          ?negDirAward rdfs:label ?negDirAwardLabel .
          FILTER(LANG(?negDirAwardLabel) = "en" && CONTAINS(LCASE(?negDirAwardLabel), "raspberry"))
        }}
      }}
      OPTIONAL {{
        ?movie wdt:P136 ?genre.
        ?genre rdfs:label ?genreLabel. FILTER(LANG(?genreLabel) IN ("en","mul")) 
      }}
    }}
    GROUP BY ?movie
    """

    sparql.setQuery(sparql_query)

    # Hálózati hívás indítása a Wikidata felé, majd a JSON válasz feldolgozása.
    # Sikeres lekérdezés esetén kibontjuk az első találat mezőit.
    sparql.setTimeout(60)
    sparql.setMethod("POST")
    results= None
    last_error = None
    for attempt in range(5):
        try:
            results = sparql.query().convert()
            break
        except Exception as e:
            last_error = e
            if attempt < 4:
                wait = 2 ** (attempt+1)
                print(f" ! {movie_title} ({target_year}): {e}, újraprobálom {wait} s múlva ({attempt+1}/5) ")
                time.sleep(wait)

    if results is None:
        print(f" !Hiba a {movie_title} ({target_year}) lekérdezésekor: {last_error}")
        return {"error": str(last_error)}

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    row = bindings[0]
    film_neg = int(row.get("filmAwardsNeg", {}).get("value", "0"))
    dir_neg = int(row.get("directorAwardsNeg", {}).get("value", "0"))
    return {
        "wikidata_uri": row.get("movie", {}).get("value"),
        "director_name" : row.get("dirName", {}).get("value", "Unknown"),
        "director_awards_pos" : int(row.get("directorAwardsPos", {}).get("value", "0")),
        "director_awards_neg" : dir_neg,
        "film_awards_pos" : int(row.get("filmAwardsPos", {}).get("value", "0")),
        "film_awards_neg" : film_neg,
        "has_neg_award": 1 if (film_neg > 0 or dir_neg > 0) else 0,
        "genres" : row.get("genres", {}).get("value", ""),
    }


if __name__ == "__main__":
    test_result = fetch_movie_data("Toy Story 4", 2019)
    print("\nEredmény a Wikidata-ról:")
    print(test_result)
