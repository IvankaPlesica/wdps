import re
import requests
from llama_cpp import Llama
import spacy

# Load SpaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# SPARQL endpoint for querying Wikidata
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Specify and initialize LLM
model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

# Regular expression patterns for matching different types of entities
patterns = {
    'Entity': re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'),  # Matches capitalized words or multi-word names
    'Abbreviation': re.compile(r'\b([A-Z]{2,})(?:\.[A-Z]+)*\b')  # Matches abbreviations like U.S., NASA, U.K. and USA, OK (probably)
    # Not broad enough and should include more cases
    # Either that or replace with some other method, maybe Byte Pair Encoding
}

# List of common words (stopwords) that shouldn't be considered entities
# The problem with entites as King of England; not a problem for entites like King Of England
# I read that built-in models have their own stop-words but I'm not sure if we're allowed to use those
stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
             "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
             "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
             "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
             "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
             "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
             "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
             "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

# Tokenization: splits text into words while keeping multi-word entities intact
def tokenize(text):
    return re.findall(r'\b[A-Z][a-z]+(?:[\w\']*\b)+', text)

# Function for simple entity recognition
def recognize_entities(text):
    entities = {}

    # Search for matches in the text using patterns
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if match.lower() not in stopwords:
                entities[match] = True

    # Tokenize the text and check for capitalized words that might be entities
    tokens = tokenize(text)
    for token in tokens:
        if token.istitle() and token.lower() not in stopwords and token not in entities:
            entities[token] = True

    # Remove substrings and filter entities
    entities_list = list(entities.keys())
    entities_list.sort(key=len, reverse=True)  # Sort entities by length (longer entities first); might delete later

    filtered_entities = {}
    for entity in entities_list:
        if not any(entity in longer_entity for longer_entity in filtered_entities):
            filtered_entities[entity] = True

    return filtered_entities

# Disambiguate entities using spaCy
def disambiguate_entities(entities, text):
    doc = nlp(text)
    disambiguated_entities = {}

    for ent in doc.ents:
        if ent.text in entities:
            disambiguated_entities[ent.text] = ent.label_  # Label the entity 'PERSON', 'ORG',...

    return disambiguated_entities

# Query Wikidata for a given entity text; LIMIT 1 so it only returns 1 query; maybe too slow and queries should be handeled in batches
def query_wikipedia_link(entity_text):
    sparql_query = f"""
    SELECT ?sitelink WHERE {{
      ?item rdfs:label "{entity_text}"@en.
      ?sitelink schema:about ?item;
               schema:isPartOf <https://en.wikipedia.org/>.
    }}
    LIMIT 1
    """
    response = requests.get(SPARQL_ENDPOINT, params={'query': sparql_query, 'format': 'json'})
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    if results:
        return results[0]["sitelink"]["value"]
    return None

# Process input questions and generate output; files open for each entity which isn't efficient
def process_questions(input_file, output_file):
    with open(input_file, "r") as file:
        for line in file:
            id_question, question = line.strip().split("\t")
            
            # Generate response using LLM
            output = llm(question, max_tokens=32, echo=True)
            response = output['choices'][0]['text']
            
            # Extract entities using custom method
            extracted_entities = recognize_entities(response)
            disambiguated_entities = disambiguate_entities(extracted_entities, response)
            
            # Retrieve Wikipedia links dynamically and filter out entities not found
            wikipedia_links = {entity: query_wikipedia_link(entity) for entity in disambiguated_entities}
            wikipedia_links = {entity: link for entity, link in wikipedia_links.items() if link is not None}
            
            # Placeholder for extracted answer and correctness tagging
            extracted_answer = "Extracted Answer"
            expected_answer = "Expected Answer"
            correctness_tag = "Correct" if extracted_answer == expected_answer else "Incorrect"
            
            # Write results to output file
            with open(output_file, 'a') as f:
                f.write(f"{id_question}\tR\"{question} {response.strip()}\"\n")
                f.write(f"{id_question}\tA\"{extracted_answer}\"\n")
                f.write(f"{id_question}\tC\"{correctness_tag}\"\n")
                for entity, link in wikipedia_links.items():
                    f.write(f"{id_question}\tE\"{entity}\"\t\"{link}\"\n")

# Files
input_file = "input_file.txt"  
output_file = "output_file.txt"

# Process the input file and generate output
process_questions(input_file, output_file)
