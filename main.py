# from tqdm import tqdm
# from helper.document import DocumentLoader 
# from database.vectordb import Vectorstore
# from embedding.embeddings import MPNet
from hyperparameter.testing import RetrieverHyperparameter
# from langchain_core.documents import Document

def main():
    test = RetrieverHyperparameter()
    test.test()
    test.get_best_params()
    
if __name__ == '__main__':
    main()