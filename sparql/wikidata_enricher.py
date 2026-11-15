import sys
import time
from pathlib import Path
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, XSD

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "RottenTomatoesHybridML/1.0 (BME VIK Project)"

#Saját RDF névtér definiálása a lokális tudásgráfhoz
EX_NAMSPACE = Namespace("http://example.org/rotten_tomatoes/")

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
           (Count(DISTINCT ?dirAward) AS ?directorAwards)
           (Count(DISTINCT ?movieAward) AS ?filmAwards)
           (GROUP_CONCAT(DISTINCT ?genreLabel; SEPARATOR=", ") AS ?genres)
    WHERE {{
      ?movie wdt:P31 wd:Q11424;
             rdfs:label "{safe_title}"@en.
      ?movie wdt:P577 ?date.
      BIND(YEAR(?date) AS ?releaseYear)
      {year_filter}
      OPTIONAL {{ ?movie wdt:P166 ?movieAward. }}
      OPTIONAL {{
        ?movie wdt:P57 ?director.
        ?director rdfs:label ?directorLabel. FILTER(LANG(?directorLabel) = "en") 
        OPTIONAL {{ ?director wdt:P166 ?dirAward. }}
      }}
      OPTIONAL {{
        ?movie wdt:P136 ?genre.
        ?genre rdfs:label ?genreLabel. FILTER(LANG(?genreLabel) = "en") 
      }}
    }}
    GROUP BY ?movie
    LIMIT 1
    """

    sparql.setQuery(sparql_query)

    # Hálózati hívás indítása a Wikidata felé, majd a JSON válasz feldolgozása.
    # Sikeres lekérdezés esetén kibontjuk az első találat mezőit.
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        row = bindings[0]
        return {
            "wikidata_uri": row.get("movie", {}).get("value"),
            "director_name" : row.get("dirName", {}).get("value", "Unknown"),
            "director_awards" : int(row.get("directorAwards", {}).get("value", "0")),
            "film_awards" : int(row.get("filmAwards", {}).get("value", "0")),
            "genres" : row.get("genres", {}).get("value", ""),
        }
    except Exception as e:
        print(f" !Hiba a {movie_title} ({target_year}) lekérdezésekor: {e}")
        return None
if __name__ == "__main__":
    test_result = fetch_movie_data("Batman v Superman", "2016")
    print("\nEredmény a Wikidata-ról:")
    print(test_result)
