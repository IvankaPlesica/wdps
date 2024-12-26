import argparse
import multiprocessing
import time
import spacy
import os
from llama_cpp import Llama
from datetime import datetime
from entity_extractor import extract_entities
from answer_processor import AnswerProcessor

model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"
nlp = spacy.load("en_core_web_sm")

def log(message):
    """Log with timestamp and process ID."""
    print(f"{datetime.now()} [PID: {os.getpid()}] {message}")

def process_tasks(question_data):
    id_question, question = question_data
    start_time = datetime.now()
    log(f"Starting question: {id_question} at {start_time}")
    
    
    llm = Llama(model_path=model_path, verbose=False) 
    output = llm(question, max_tokens=50, temperature=0.0, top_k=5, top_p=0.1, echo=True)
    response = output['choices'][0]['text']

    extracted_entities = extract_entities(response)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    log(f"Finished question: {id_question} at {end_time} (Duration: {duration:.2f}s)")

    return id_question,question,response, extracted_entities

def clean_response(response: str) -> str:
    unwanted_phrases = ["nobody knows", "I am not sure", "I don't know"]
    
    for phrase in unwanted_phrases:
        response = response.replace(phrase, "").strip()
    
    return response

def process_questions_parallel(input_file, output_file, num_processes=2):
    with open(input_file, "r") as file:
        lines = file.readlines()
    
    question_data = []
    for line in lines:
        id_question, question = line.strip().split("\t")
        question_data.append((id_question, question))
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.imap_unordered(process_tasks, question_data))    
    
    results.sort(key=lambda x: x[0])
    processor = AnswerProcessor()

    with open(output_file, "w") as f:
        for id_question, question, response, extracted_entities in results:
            entities = {entity: link for entity, link in extracted_entities.items()}
            context = clean_response(response)
            
            result = processor.process_question(question, context, entities)
            extracted_answer = result['answer']
            validation = result['validation']
            correctness = "correct" if validation['is_valid'] else "incorrect"
            
            f.write(f"{id_question}\tR\"{response.strip()}\"\n")
            f.write(f"{id_question}\tA\"{extracted_answer}\"\n")
            f.write(f"{id_question}\tC\"{correctness}\"\n")
            for entity, link in extracted_entities.items():
                f.write(f"{id_question}\tE\"{entity}\"\t\"{link}\"\n")

    print("Processing and writing complete.")

if __name__ == "__main__":
    #Set up argument parsing
    parser = argparse.ArgumentParser(description="Process questions using Llama model.")
    parser.add_argument("input_file", type=str, help="Path to the input file containing questions.")
    parser.add_argument("output_file", type=str, help="Path to the output file to save results.")
    parser.add_argument("--num_processes", type=int, default=2, help="Number of processes to use for parallel processing.")
    args = parser.parse_args()

    # Extract arguments
    input_file = args.input_file
    output_file = args.output_file
    num_processes = args.num_processes

    # Start processing
    print("Start processing...")
    start_time = time.time()
    process_questions_parallel(input_file, output_file, num_processes)
    end_time = time.time()

    # Print runtime
    runtime = end_time - start_time
    print(f"Processing complete. Total runtime: {runtime:.2f} seconds.")
