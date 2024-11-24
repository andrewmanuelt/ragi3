from langchain_huggingface import HuggingFaceEmbeddings

class MPNet():
    def __init__(self):
        self.model_name = 'mpnet'
        self.model_repo = 'sentence-transformers/all-mpnet-base-v2'
        
    def load_embedding(self):
        return HuggingFaceEmbeddings(model_name=self.model_repo)