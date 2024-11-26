import json 

from tqdm import tqdm
from database.vectordb import Vectorstore
from helper.document import DocumentLoader
from embedding.embeddings import MPNet

class RetrieverHyperparameter():
    def _param(self):
        param = {
            'test_name': 'dummy',
            'chunk_size': [300, 500],
            'chunk_overlap': [0, 30], 
            'top_k': [3],
            'dataset_train_dir': './dataset/dummy/dummy.json', 
            'dataset_test_dir': './dataset/dummy/dummy_test.json', 
        }
        
        return param['test_name'], param['chunk_size'], param['chunk_overlap'], param['top_k'], param['dataset_train_dir'], param['dataset_test_dir']
    
    def run(self):
        test_name, chunk_size, chunk_overlap, top_k, train_dir, test_dir = self._param()
        
        num_documents = []
        test_question_list = self._prepare_test(test_dir)
        
        print(" starting training dataset")
        
        for cz in tqdm(chunk_size):
            for co in tqdm(chunk_overlap):
                for tk in tqdm(top_k):
                    num_document = self._prepare_train(
                        test_name=test_name,
                        chunk_size=cz, 
                        chunk_overlap=co,
                        train_dir=train_dir,
                        top_k=tk
                    )
                    
                    num_documents.append(num_document)
        
        loop = 0
        
        result_collection = []          
        for cz in tqdm(chunk_size):
            for co in chunk_overlap:
                for tk in tqdm(top_k):
                    result, mean_score = self._do_test(
                        test_name=test_name,
                        chunk_size=cz,
                        chunk_overlap=co,
                        top_k=tk,
                        test_question_list=test_question_list
                    )
                    
                    row = {
                        'chunk_size': cz, 
                        'chunk_overlap': co, 
                        'top_k': tk,
                        'mean_score': mean_score,
                        'number_of_train_documents': num_documents[loop],
                        'result': result,
                    }
                    
                    loop = loop + 1
                    
                    result_collection.append(row)
        
        with open(f'./hyperparameter/result_{test_name}.json', 'w') as f:
            json.dump(result_collection, f, indent=4)
                        
    def _do_test(self, test_name, chunk_size, chunk_overlap, top_k, test_question_list):
        embedding = MPNet()
        embedding = embedding.load_embedding()
        
        store = Vectorstore(
            embedding_function=embedding, 
            index_name=f"{test_name}_{chunk_size}_{chunk_overlap}_{top_k}",
            local_path=f"./database/{test_name}/{chunk_size}_{chunk_overlap}_{top_k}"
        )
        client = store.load_localfile()
        
        collection_result = []
        score = 0
        for test_question in test_question_list: 
            result, mean_score = store.search(
                client=client, 
                query=test_question,
                top_k=top_k
            )
            
            collection_question = {
                'query': test_question,
                'result': result
            }
            collection_result.append(collection_question)

            score = score + mean_score
        return collection_result, score / len(test_question_list)
        
    def _prepare_train(self, test_name, chunk_size, chunk_overlap, train_dir, top_k):
        embedding = MPNet()
        embedding = embedding.load_embedding()
        
        # db preparation
        store = Vectorstore(
            embedding_function=embedding, 
            index_name=f"{test_name}_{chunk_size}_{chunk_overlap}_{top_k}",
            local_path=f"./database/{test_name}/{chunk_size}_{chunk_overlap}_{top_k}"
        )
        client = store.client()
        
        # document preparation
        loader = DocumentLoader(
            dataset_path=train_dir
        )
        documents = loader.document()
        
        num_document = 0
        for doc in tqdm(documents):
            chunked_test = loader.chunk_context(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                context=doc['context']
            )
            
            for small_chunk in chunked_test:
                num_document = num_document + 1
                
                doc = loader.chunked_to_json_item(
                    chunked_test=small_chunk, 
                    question=doc['question'], 
                    answer=doc['answer']
                )

                document = loader.json_to_document(document=doc)

                store.store_document(
                    client=client, 
                    document=document
                )
                store.store_localfile(client)

        return num_document
    
    # done
    def _prepare_test(self, test_dir) -> list:
        collection = []
        
        loader = DocumentLoader(
            dataset_path=test_dir
        )
        
        document = loader.document()
        
        for row in document:
            collection.append(row['question'])
            
        return collection        
    
    def get_best_params(self):
        with open(f'./hyperparameter/result_{self._param()[0]}.json') as f:
            data = json.load(f)
        
        temp = 0
        for index, row in enumerate(data):
            if row['mean_score'] > temp:
                temp = index 
        
        print("best params: ")
        print(f"* chunk size: {data[index]['chunk_size']} ")
        print(f"* chunk overlap: {data[index]['chunk_overlap']} ")
        print(f"* top k: {data[index]['top_k']} ")
        print(f"* mean score: {data[index]['mean_score']} ")