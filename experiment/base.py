import json 
import requests

from abc import ABC, abstractmethod
from helper.document import DocumentLoader
from evaluator.evaluate import Evaluator

class BaseExperiment(ABC):
    def __init__(self):
        self.dataset_test_dir = ''
        self.index_name = ''
        self.model_name = ''
        self.model_repo = ''
        self.temperature = 0
        self.repetition_penalty = 0
    
    @abstractmethod
    def _prepare_test(self) -> list:
        pass 
    
    @abstractmethod
    def run(self):    
        pass 
    
    @abstractmethod 
    def _prompt(self, question: str, context: str) -> str: 
        pass 
    
    @abstractmethod 
    def _api_request(self):
        pass 
    
    @abstractmethod
    def _evaluation_score(self, candidate: str, reference: str) -> float:
        pass
    
class KomodoExperiment(BaseExperiment):
    def _prepare_test(self) -> list:
        collection = [] 
        
        loader = DocumentLoader(
            dataset_path=self.dataset_test_dir
        )
        document = loader.document()
        
        for row in document:
            qa_set = (row['question'], row['answer'])
            collection.append(qa_set)
        
        return collection
    
    def run(self):
        test_dataset = self._prepare_test()
        
        collection = []
        for question, groundtruth in test_dataset:
            context = self._get_context_from_retriever(question)
            prompt = self._prompt(question, context)
            candidate = self._api_request(question, prompt)
            
            meteor, bertscore, rouge = self._evaluation_score(candidate, groundtruth)
            row = {
                'question': question, 
                'candidate': candidate,
                'reference': groundtruth,
                'meteor': meteor, 
                'bertscore': bertscore, 
                'rouge': rouge
            }
            
            collection.append(row)
        
        with open(f'./result_{str(self.model_name).lower()}.json', 'w') as f:
            json.dump(collection, f, indent=4)
         
    def _get_context_from_retriever(self, question):
        pass
    
    def _prompt(self, question: str, context: str) -> str: 
        with open('./prompt/prompt_ext.txt') as f:
            prompt = f.read()
        
        return prompt.format(question=question, context=context)

    def _api_request(self, prompt):
        url = 'http://127.0.0.1:8000/v1/chat/completions'
        
        headers = {
            'Accept'
            'Content-Type': 'application/json',
        }

        content = f"Please answer the question below in Bahasa Indonesia. {prompt}"
        
        body = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant.",
                },
                {
                    "role": "user",
                    "content": f"{content}",
                }
            ],
            "temperature": 0.8,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "max_tokens": 100,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {}
        }

        response = requests.post(
            url=url,
            headers=headers, 
            data=json.dumps(body),
        )
        
        return response.text

    def _evaluation_score(self, candidate: str, reference: str) -> float:
        evaluation = Evaluator()
        
        meteor = evaluation.meteor(candidate, reference)
        bertscore = evaluation.bertscore(candidate, reference)
        rouge = evaluation.rouge(candidate, reference)

        return meteor, bertscore, rouge

    
