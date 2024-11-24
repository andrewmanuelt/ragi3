import sys
import faiss 
import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)

from uuid import uuid4

from langchain.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

class Vectorstore():
    def __init__(self, local_path, index_name, embedding_function):
        self.local_path = local_path
        self.index_name = index_name
        self.embedding_function = embedding_function
    
    def client(self):
        index = faiss.IndexFlatL2(len(self.embedding_function.embed_query("vectorstore")))
        
        client = FAISS(
            docstore = InMemoryDocstore(),
            embedding_function = self.embedding_function,
            index = index, 
            index_to_docstore_id = {}
        )
        
        return client
    
    def store_document(self, client, document):
        id = str(uuid4())
        
        client.add_documents(
            documents=[document], 
            ids=[id]
        )
        
    def store_localfile(self, client):
        client.save_local(
            folder_path=self.local_path, 
            index_name=self.index_name
        )
    
    def load_localfile(self):
        client = FAISS.load_local(
            folder_path=self.local_path, 
            index_name=self.index_name, 
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True
        )
        
        return client
    
    def search(self, client, query, top_k):
        collection = []
        results = client.similarity_search_with_score(query=query, k=top_k)
        
        mean_score = 0
        
        for result in results: 
            row = {
                'context': result[0].page_content,
                'distance': str(result[1]),
            }
            
            collection.append(row)
            mean_score = mean_score + result[1]

        return collection, mean_score / top_k