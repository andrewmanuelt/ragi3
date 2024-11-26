from evaluator.evaluate import Evaluator

from hyperparameter.testing import RetrieverHyperparameter
from experiment.base import KomodoExperiment

def main():
    # test = RetrieverHyperparameter()
    # test.run()
    # test.get_best_params()
    
    # candidate = "hello, i'm gemma"
    # reference = "hello fellas we are gemma!"
    # evaluation = Evaluator()
    # evaluation.meteor(candidate, reference)
    # evaluation.bertscore(candidate, reference)
    # evaluation.rouge(candidate, reference)

    komodo = KomodoExperiment()
    prompt = komodo._prompt("who is obama?", "Barack Hussein Obama II is an American lawyer and politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African-American president in U.S. history.")
    response = komodo._api_request(prompt)

    print(response)
    
if __name__ == '__main__':
    main()