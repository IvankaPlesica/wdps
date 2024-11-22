# 1 - PART
#--------------------------------------------------------------------------------------------------------------------
# Import LLM 
# We are required to use this language model.
from llama_cpp import Llama

# Import NLP, but only temporarily. This is the part that needs to be implemented.
# In order for this to work, it needs to be installed in the container.
# pip install spacy
# python -m spacy download en_core_web_sm
# This block will be deleted.
import spacy
nlp = spacy.load("en_core_web_sm")

# Specify and initialize the LLM path
model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

# We expect that the input of our program is a text file of the form 
# <ID question><TAB>text of the question/completion<newline>
# so to test this, we have a file input_file.txt with some questions. For example:
# question-001  Who wrote a song 'Imagine'?
# question-002  On what continent is The Netherlands?
# Open the input file
with open("input_file.txt", "r") as file:
    for line in file:
            id_question, question = line.strip().split("\t")

# Generate the output for the question by calling LLM on input question
output = llm(
      question, 
      max_tokens=32, # can be experimented with; it is the size of the text that will be generated
      #stop=["Q", "\n"],   If I leave this in, it doesn't generate anything beyond adding a question to the output
      echo=True # True so it includes the question in the output. We want this so it creates entites for questions too.
)

# Extract the LLM's response
# The response is a dictionary, but we're only interested in the text it generated so we choose only the first 
# element of the dictionary.
response = output['choices'][0]['text']

# 2 - PART
# -----------------------------------------------------------------------------------------------------------------
# Again, this is only a temporary way of extracting entities. We use it here to illustrate what we're expected to do.
# This one line extracts entites from a generated text.
# There are many ways of extracting entites explained in the learning materials. We have to choose some and implement.
# Our Task 1 is to do the following single line of code. Extracted, expected answer and a correctness are a not a part
# of Task 1. The second part of the task is at the end of 3 - PART
doc = nlp(response)

# Placeholder for answer extraction (we will replace this later with actual logic)
extracted_answer = "Extracted Answer"  # Replace with logic for extracting answers

# Placeholder for correctness tagging (replace with actual check logic)
expected_answer = "Expected Answer"  # Replace with actual expected answer
correctness_tag = "Correctness Answer" if extracted_answer == expected_answer else "incorrect"


# 3 - PART
# ------------------------------------------------------------------------------------------------------------------
# Create the output and write to a file 
# Your output should be a file where the answer are in the following format
# <ID question><TAB>[R,A,C,E]<answer> where 
# "R" indicates the raw text produced by the language model, 
# "A" is the extracted answer, 
# "C" is the tag correct/incorrect and 
# "E" are the entities extracted.
with open('output_file.txt', 'a') as f:
# Write raw model response
f.write(f"{id_question}\tR\"{response.strip()}\"\n")
            
# Write extracted answer
f.write(f"{id_question}\tA\"{extracted_answer}\"\n")
            
# Write correctness tag
f.write(f"{id_question}\tC\"{correctness_tag}\"\n")
            
# Task 1 
# Write entities and their corresponding Wikipedia links (using placeholders for now)
for ent in doc.ents:
    # Placeholder Wikipedia links (We will replace this with actual logic later)
    # It can't be literal like this because we can have a case like this:
    # Italian Kingdom ⇒ https://en.wikipedia.org/wiki/Kingdom_of_Italy
    # I think it has to be done through a KB, but I'm not actually sure.
    # Maybe we group them together earlier so we don't have to think about it here. 
    # We were advised to use Wikidata, so I'll assume we will.
    wikipedia_link = f"https://en.wikipedia.org/wiki/{ent.text.replace(' ', '_')}"
    f.write(f"{id_question}\tE\"{ent.text}\"\t\"{wikipedia_link}\"\n")





