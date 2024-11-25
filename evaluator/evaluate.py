import re 
import evaluate

from bert_score import BERTScorer

from nltk.translate.meteor_score import meteor_score

class Evaluator():
    def meteor(self, candidate: str, reference: str) -> float:
        candidate = self._remove_punctuation(candidate).split()
        reference = self._remove_punctuation(reference).split()
        
        return meteor_score([candidate], reference)
    
    def bertscore(self, candidate: str, reference: str):
        candidate = [self._remove_punctuation(candidate)]
        reference = [self._remove_punctuation(reference)]
        
        score = BERTScorer(model_type='bert-base-uncased')
        prec, recl, f1 = score.score(candidate, reference)
        
        return prec.mean(), recl.mean(), f1.mean()
    
    def rouge(self, candidate: str, reference: str):
        rougescore = evaluate.load('rouge')
        
        candidate = [self._remove_punctuation(candidate)]
        reference = [self._remove_punctuation(reference)]
        
        score = rougescore.compute(
            predictions=candidate, 
            references=reference, 
        )
        
        return score['rougeL']
        
    def _remove_punctuation(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()