from evaluator.evaluate import Evaluator

from hyperparameter.testing import RetrieverHyperparameter

def main():
    # test = RetrieverHyperparameter()
    # test.run()
    # test.get_best_params()
    
    candidate = "hello, i'm gemma"
    reference = "hello fellas we are gemma!"
    
    evaluation = Evaluator()
    evaluation.meteor(candidate, reference)
    evaluation.bertscore(candidate, reference)
    evaluation.rouge(candidate, reference)
        
    
if __name__ == '__main__':
    main()