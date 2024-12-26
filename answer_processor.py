# answer_processor.py
import spacy
from transformers import pipeline
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class QuestionAnalysis:
    question_type: str  # 'yes_no' or 'entity'
    expected_entity_type: Optional[str] = None
    target_entity: Optional[str] = None
    property_type: Optional[str] = None

class AnswerProcessor:
    def __init__(self):
        """Initialize the answer processor with models and configurations."""
        # Load SpaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load RoBERTa question-answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",  # Fine-tuned RoBERTa model
            tokenizer="deepset/roberta-base-squad2",
            device=-1 
        )
        
        # Patterns for question type detection
        self.yes_no_patterns = [
            r'^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\s',
            r'^(?:isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|hasn\'t|haven\'t|hadn\'t)\s'
        ]
        
        # Keywords for answer detection
        self.yes_indicators = ['yes', 'correct', 'true', 'right', 'indeed', 'surely']
        self.no_indicators = ['no', 'incorrect', 'false', 'wrong', 'not']

    def analyze_question(self, question: str) -> QuestionAnalysis:
        """Analyze the question to determine its type and expected answer properties."""
        question = question.lower().strip()
        
        # Check for yes/no question
        for pattern in self.yes_no_patterns:
            if re.match(pattern, question):
                doc = self.nlp(question)
                entities = [ent.text for ent in doc.ents]
                return QuestionAnalysis(
                    question_type='yes_no',
                    target_entity=entities[0] if entities else None
                )
        
        # Entity question analysis
        doc = self.nlp(question)
        question_word = next((token.text for token in doc if token.tag_ in ['WP', 'WRB']), None)
        
        if question_word:
            entity_type = None
            if question_word == 'who':
                entity_type = 'PERSON'
            elif question_word == 'where':
                entity_type = 'LOCATION'
            elif question_word == 'when':
                entity_type = 'DATE'
            
            return QuestionAnalysis(
                question_type='entity',
                expected_entity_type=entity_type
            )
        
        return QuestionAnalysis(question_type='entity')

    def extract_answer(self, question: str, context: str) -> Dict:
        """
        Extract the answer using RoBERTa QA pipeline.
        
        Args:
            question (str): The user's question.
            context (str): The context for the QA pipeline.

        Returns:
            Dict: Extracted answer and confidence score.
        """
        try:
            qa_result = self.qa_pipeline(question=question, context=context)
            answer = qa_result['answer']
            score = qa_result['score']
            
            return {'answer': answer, 'confidence': score}
        except Exception as e:
            print(f"Error in QA pipeline: {e}")
            return {'answer': None, 'confidence': 0.0}

    def validate_answer(self, answer: Dict, entities: Dict[str, str]) -> Dict:
        """Validate the extracted answer using entities."""
        validation_result = {
            'is_valid': False,
            'confidence': 0.0,
            'source': None
        }
        
        try:
            if answer['answer'] in entities:
                validation_result = {
                    'is_valid': True,
                    'confidence': answer['confidence'],
                    'source': 'entity_match'
                }
        except Exception as e:
            print(f"Validation error: {e}")
            
        return validation_result

    def process_question(self, question: str, context: str, entities: Dict[str, str]) -> Dict:
        """Process a single question through the pipeline."""
        try:            
            answer_data = self.extract_answer(question, context)
            
            validation_result = self.validate_answer(answer_data, entities)
            
            return {
                'question': question,
                'answer': answer_data['answer'],
                'confidence': answer_data['confidence'],
                'validation': validation_result
            }
        except Exception as e:
            print(f"Error processing question: {e}")
            return {'error': str(e)}