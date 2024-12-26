import argparse
import multiprocessing
import os
import requests
import spacy
import time
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
from llama_cpp import Llama
from datetime import datetime

from answer_processor import AnswerProcessor

model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"
nlp = spacy.load("en_core_web_sm")

def log(message):
    """Log with timestamp and process ID."""
    print(f"{datetime.now()} [PID: {os.getpid()}] {message}")

# Function to fetch candidate entities from Wikidata based on a label
# Fetch function for candidates
def fetch_candidate(label):
    search_label = remove_leading_articles(label)

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": search_label,
        "language": "en",
        "limit": 15,
        "format": "json",
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return [
            {
                "id": item["id"],
                "label": item["label"],
                "title": item["label"],  
                "description": item.get("description", ""),
                "aliases": item.get("aliases", ""),
            }
            for item in data.get("search", [])
        ]
    except Exception as e:
        print(f"Error fetching candidate for {label}: {e}")
        return []

import requests

def fetch_batch_entity_aliases(candidate_ids):
    """Fetch aliases, descriptions, and popularity indicators (sitelinks, claims) for multiple candidates."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(candidate_ids),  # Combine IDs with '|'
        "props": "aliases|descriptions|sitelinks|claims",
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        entity_data = {}
        for entity_id in candidate_ids:
            entity_info = data["entities"].get(entity_id, {})

            alias_values = [alias["value"] for alias in entity_info.get("aliases", {}).get("en", [])]
            
            description = entity_info.get("descriptions", {}).get("en", {}).get("value", "")
            
            sitelinks = entity_info.get("sitelinks", {})

            sitelinks_count = len(entity_info.get("sitelinks", {}))
            
            claims_count = sum(len(entity_info.get("claims", {}).get(claim, [])) for claim in entity_info.get("claims", {}))

            entity_data[entity_id] = {
                "aliases": alias_values,
                "description": description,
                "sitelinks": sitelinks,
                "sitelinks_count": sitelinks_count,
                "claims_count": claims_count
            }

        return entity_data

    except Exception as e:
        print(f"Error fetching data for IDs {candidate_ids}: {e}")
        return {}

# Function to fetch the Wikipedia link for a given Wikidata ID
def wikidata_to_wikipedia(entity_data, lang="en"):
    """
    Extract Wikipedia links from existing sitelinks data.

    Args:
        entity_data (dict): A dictionary containing entities' data, including sitelinks.
        lang (str): Language code for Wikipedia (default is "en").

    Returns:
        dict: A dictionary with Wikidata IDs as keys and Wikipedia URLs as values.
    """
    try:
        sitelinks = entity_data['candidate']['sitelinks']

        if f"{lang}wiki" in sitelinks:
            title = sitelinks[f"{lang}wiki"]["title"]
            return f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    except KeyError:
        return None
    
def deduplicate_entities(entities):
    seen = set()
    deduplicated = []
    
    for entity in entities:
        normalized = entity['label'].strip().lower()  # Normalize: trim and convert to lowercase
        if normalized not in seen:
            deduplicated.append(entity)  # Append the original entity to preserve casing
            seen.add(normalized)
    
    return deduplicated

def score_similarity(str1, str2):
    score = SequenceMatcher(None, str1, str2).ratio()
    return score

def normalize_text(text):
    doc = nlp(text.lower())
    normalized = " ".join([token.lemma_ for token in doc if token.is_alpha])
    return normalized

def remove_leading_articles(text):
    articles = ["the ", "a ", "an "]
    for article in articles:
        if text.lower().startswith(article):
            return text[len(article):].strip()
    return text

def rank_candidates_by_similarity(candidates, context, entity):
    context_norm = normalize_text(context)

    ranked_by_context = []
    ranked_by_total = []

    for candidate in candidates:

        title = candidate['label']
        aliases = candidate.get("aliases", [])  # Fetch aliases, default to an empty list
        sitelinks_count=candidate['sitelinks_count']
        claims_count = candidate['claims']

        # Compute title match score
        if title.lower() == entity['label'].lower() or title.lower() == remove_leading_articles(entity['label']).lower():
            title_match_score = 1.0
        elif any(alias.lower() == entity['label'].lower() for alias in aliases):
            title_match_score = 0.95  # Match with an alias
        else:
            title_match_score = score_similarity(entity['label'], title)

        # Context similarity
        context_similarity = cosine_similarity(candidate['description'].lower(), context_norm.lower())

        sitelinks_weight = sitelinks_count / 100.0
        claims_weight = claims_count / 50.0 if claims_count else 0

        total_similarity = (
            0.6 * title_match_score +
            0.5 * context_similarity +
            0.6 * sitelinks_weight +  
            0.6 * claims_weight  
        )

        ranked_by_context.append({
            "candidate": candidate,
            "similarity": 0.5 * context_similarity + 0.6 * title_match_score,
        })

        ranked_by_total.append({
            "candidate": candidate,
            "similarity": total_similarity,
        })

    # Sort candidates by similarity
    ranked_by_context.sort(key=lambda x: x['similarity'], reverse=True)
    ranked_by_total.sort(key=lambda x: x['similarity'], reverse=True)

    best_candidate = ranked_by_context[0]
    wiki_link = wikidata_to_wikipedia(best_candidate)
    if not wiki_link:
        best_candidate = ranked_by_total[0]
        wiki_link = wikidata_to_wikipedia(best_candidate)
        
    return wiki_link



def score_similarity(str1, str2):
    """Score the similarity between two strings using difflib's SequenceMatcher."""
    score = SequenceMatcher(None, str1, str2).ratio()
    return score

def cosine_similarity(candidate_description, context):
    candidate_description_norm = normalize_text(candidate_description)
    
    text = [candidate_description_norm, context]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text)

    similarity_score = cs(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_score[0][0]

def add_aliasses_to_candidates(candidates):
    candidate_ids = [candidate["id"] for candidate in candidates]
    #entity_data = fetch_batch_entity_aliases(candidate_ids)
    aliases = fetch_batch_entity_aliases(candidate_ids)

    for candidate in candidates:
        metadata = aliases.get(candidate["id"], {})
        
        candidate["aliases"] = metadata.get("aliases", []) 
        candidate["description"] = metadata.get("description", []) 
        candidate["sitelinks"] = metadata.get("sitelinks", []) 
        candidate["sitelinks_count"] = metadata.get("sitelinks_count", []) 
        candidate["claims"] = metadata.get("claims_count", []) 

    return candidates

def fetch_candidates(label):
    candidates_enitites = fetch_candidate(label)
    candidates = add_aliasses_to_candidates(candidates_enitites)

    return candidates

def extract_entities(response):
    doc = nlp(response)
    entities = [
            {"label": ent.text, "type": ent.label_, "context": ent.sent.text}
            for ent in doc.ents
    ]

    entities = deduplicate_entities(entities)
    
    linked_entities = {}

    for entity in entities:
        candidates = fetch_candidates(entity['label'])
        wiki_link = rank_candidates_by_similarity(candidates, response, entity)
        linked_entities[entity['label']] = wiki_link

    return linked_entities