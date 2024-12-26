# Question Processing Pipeline

This project processes questions using advanced NLP techniques and machine learning models, including LLama (Llama-2-7b), SpaCy, and RoBERTa. It supports parallel processing and CLI-based execution for efficient task handling.

---

## **Installation**

To set up and run the project, follow these steps:

### **1. Ensure Requirements are Met**
Before running the project, ensure the following:
1. The necessary dependencies are installed. These are listed in `requirements.txt` and include:
   - `spacy`
   - `transformers`
   - `llama-cpp-python`
   - `scikit-learn`
   - `requests`
   

### **2. Running the code**
The script supports CLI arguments for specifying the input file, output file, and the number of processes for parallel execution. The format is:

    python main.py input_file.txt output_file.txt --num_processes <number_of_processes>