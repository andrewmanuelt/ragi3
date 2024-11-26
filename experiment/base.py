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
        for test, groundtruth in test_dataset:
            candidate = self._api_request(test)
            
            meteor, bertscore, rouge = self._evaluation_score(candidate, groundtruth)
            row = {
                'question': test, 
                'candidate': candidate,
                'reference': groundtruth,
                'meteor': meteor, 
                'bertscore': bertscore, 
                'rouge': rouge
            }
            
            collection.append(row)
        
        with open(f'./result_{str(self.model_name).lower()}.json', 'w') as f:
            json.dump(collection, f, indent=4)
            
    def _prompt(self, question: str, context: str) -> str: 
        with open('./prompt/prompt_ext.txt') as f:
            prompt = f.read()
        
        return prompt.format(question, context)

    def _api_request(self):
        url = 'http://localhost:8000'
        
        headers = {
            'content-type': 'application/json',
        }
        
        body = {
            '': ''
        }
        
        response = requests.get(
            url=url,
            headers=headers, 
            data=body,
        )
        
        return response.text

    def _evaluation_score(self, candidate: str, reference: str) -> float:
        evaluation = Evaluator()
        
        meteor = evaluation.meteor(candidate, reference)
        bertscore = evaluation.bertscore(candidate, reference)
        rouge = evaluation.rouge(candidate, reference)

        return meteor, bertscore, rouge

    
