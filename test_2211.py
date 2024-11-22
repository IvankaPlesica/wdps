import requests
from llama_cpp import Llama
import spacy

# Load SpaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# SPARQL endpoint for querying Wikidata
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Specify and initialize the LLM path
model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

# Query Wikidata for a given entity text
def query_wikipedia_link(entity_text):
    """Query Wikidata for an entity and return its English Wikipedia link."""
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
    
    # Return None if no Wikipedia link is found (ignoring the entity)
    return None

# Extract entities using SpaCy
def extract_entities_with_spacy(response):
    """Extract named entities using SpaCy."""
    doc = nlp(response)
    return [ent.text for ent in doc.ents]

# Process input questions and generate output
def process_questions(input_file, output_file):
    """Process input questions, query LLM, extract entities, and save results."""
    with open(input_file, "r") as file:
        for line in file:
            id_question, question = line.strip().split("\t")
            
            # Generate response using LLM
            output = llm(question, max_tokens=32, echo=True)
            response = output['choices'][0]['text']
            
            # Extract entities using SpaCy
            extracted_entities = extract_entities_with_spacy(response)
            
            # Retrieve Wikipedia links dynamically and filter out entities not found
            wikipedia_links = {entity: query_wikipedia_link(entity) for entity in extracted_entities}
            wikipedia_links = {entity: link for entity, link in wikipedia_links.items() if link is not None}  # Ignore None values
            
            # Placeholder for extracted answer and correctness tagging
            extracted_answer = "Extracted Answer"  # Replace with extraction logic
            expected_answer = "Expected Answer"  # Replace with actual expected answer
            correctness_tag = "Correct" if extracted_answer == expected_answer else "Incorrect"
            
            # Write results to output file
            with open(output_file, 'a') as f:
                # Write raw response (including the input question)
                f.write(f"{id_question}\tR\"{question} {response.strip()}\"\n")
                
                # Write extracted answer
                f.write(f"{id_question}\tA\"{extracted_answer}\"\n")
                
                # Write correctness tag
                f.write(f"{id_question}\tC\"{correctness_tag}\"\n")
                
                # Write entities and links (ignore entities without links)
                for entity, link in wikipedia_links.items():
                    f.write(f"{id_question}\tE\"{entity}\"\t\"{link}\"\n")

# Files
input_file = "input_file.txt"  
output_file = "output_file.txt"

# Process the input file and generate output
process_questions(input_file, output_file)
