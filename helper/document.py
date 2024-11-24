import json 

from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader(): 
    # use 'None' just for splitting purpose
    def __init__(self, dataset_path = None):
        self.path = dataset_path
        
    def _loader(self):
        loader = JSONLoader(
            file_path=self.path,
            jq_schema=".[]",
            text_content=False
        )
        
        return loader.load()
    
    def document(self):
        collection = []
        
        documents = self._loader()
        for document in documents:
            document_object = json.loads(document.page_content)    
            
            collection.append(document_object)

        return collection
    
    def json_to_document(self, document):
        return Document(
            page_content=document['context'],
            metadata={
                'question': document['question'], 
                'answer': document['answer']
            }
        )
    
    def chunk_context(self, chunk_size, chunk_overlap, context) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len,
            is_separator_regex = False
        )
        
        return splitter.split_text(context)
    
    def chunked_to_json_item(self, chunked_test, question, answer):
        return {
            'question': question,
            'answer': answer, 
            'context': chunked_test
        }