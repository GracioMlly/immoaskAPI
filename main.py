import psycopg2
import os
from typing import Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Liste des possibilités de filtre
filters_possibilities = ["type", "price", "surface", "city", "origin"]


def filter_query_builder(path_parameters: dict[str, Any], all_filters: list[str]):

    usable_filters = [
        filter for filter in all_filters if filter in filters_possibilities
    ]
    usable_values = [
        (
            (path_parameters[filter]).lower()
            if type(path_parameters[filter]) == str
            else path_parameters[filter]
        )
        for filter in usable_filters
    ]

    if len(usable_filters) == 0:
        query = "SELECT offer_id, type, title, price, surface, localisation, description, images, source FROM offers"
        return [query, None]
    else:
        query = "SELECT offer_id, type, title, price, surface, localisation, description, images, source FROM offers WHERE "
        for index, filter in enumerate(usable_filters):
            match (filter):
                case "type":
                    query += (
                        "LOWER(type) = %s" if index == 0 else " AND LOWER(type) = %s"
                    )
                case "price":
                    query += (
                        "SPLIT_PART(price, 'XOF', 1)::int <= %s"
                        if index == 0
                        else " AND SPLIT_PART(price, 'XOF', 1)::int <= %s"
                    )
                case "surface":
                    query += (
                        "SPLIT_PART(surface, 'm²', 1)::int <= %s"
                        if index == 0
                        else " AND SPLIT_PART(surface, 'm²', 1)::int <= %s"
                    )
                case "city":
                    query += (
                        "LOWER(localisation ->> 'city') = %s"
                        if index == 0
                        else " AND LOWER(localisation ->> 'city') = %s"
                    )
                case "origin":
                    query += (
                        "LOWER(source ->> 'origin') = %s"
                        if index == 0
                        else " AND LOWER(source ->> 'origin') = %s"
                    )

        return [query, usable_values]


app = FastAPI()

# Paramétrage du CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hostname = os.getenv("DB_HOSTNAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")
hf_token = os.getenv("HF_TOKEN")

# Connexion à la base de données
# PostgreSQL
connection = psycopg2.connect(
    host=hostname, user=user, password=password, dbname=database
)

# Création d'un curseur pour l'exécution des commandes
cursor = connection.cursor(cursor_factory=RealDictCursor)

# Initialisation du client hugging_face
hf_client = InferenceClient(
    provider="auto",
    api_key=hf_token,
)

app = FastAPI()


# Endpoint pour récupérer toutes les annonces
@app.get("/offers")
def get_offers():
    cursor.execute(
        "SELECT offer_id, type, title, price, surface, localisation, description, images, source FROM offers"
    )
    result = cursor.fetchall()

    if result is not None:
        return result
    else:
        return {"message": f"Il n'y a aucune offre pour l'instant"}


# Endpoint pour récupérer les annonces selon
# une recherche sémantique
@app.get("/offers/search")
def search_offers(query: str, limit: int = 5):
    search_embedding = hf_client.feature_extraction(
        query,
        model="intfloat/multilingual-e5-large-instruct",
    )

    cursor.execute(
        "SELECT offer_id, type, title, price, surface, localisation, description, images, source FROM offers ORDER BY embedding <=> %s::vector LIMIT %s",
        [search_embedding.tolist(), limit],
    )
    result = cursor.fetchall()

    if result is not None:
        return result
    else:
        return {
            "message": f"Il n'y a aucune offre répondant à la recherche",
            "search": query,
        }


# Endpoint pour filtrer les annonces selon
# différents critères [type, price, surface, city, origin]
@app.get("/offers/filter")
def filter_offers(req: Request):
    query_params = dict(req.query_params)
    SQL_query, values = filter_query_builder(query_params, list(query_params.keys()))

    if values is not None:
        cursor.execute(SQL_query, values)
        result = cursor.fetchall()

        return result
    else:
        cursor.execute(SQL_query)
        result = cursor.fetchall()

        return result


# Endpoint pour récupérer une seule annonce selon
# son id
@app.get("/offers/{offer_id}")
def get_offer(offer_id: int):
    cursor.execute(
        "SELECT offer_id, type, title, price, surface, localisation, description, images, source FROM offers WHERE offer_id = %s",
        [offer_id],
    )
    result = cursor.fetchone()

    if result is not None:
        return result
    else:
        return {"message": f"Aucune offre portant l'identifiant {offer_id}"}
