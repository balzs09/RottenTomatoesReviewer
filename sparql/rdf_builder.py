import sys
import time
from pathlib import Path
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, XSD
from wikidata_enricher import fetch_movie_data

#Saját RDF névtér definiálása a lokális tudásgráfhoz
EX_NAMSPACE = Namespace("http://example.org/rotten_tomatoes/")