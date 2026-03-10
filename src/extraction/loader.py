import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader

def load_text_documents(path: str):
    """
    Loads documents from a given path (file or directory).
    For MVP, we use TextLoader for .txt and .md files.
    """
    if os.path.isfile(path):
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    elif os.path.isdir(path):
        # Load both .txt and .md files
        # Depending on system, it's simpler to use separate Loaders
        txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        md_loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        
        docs = []
        try:
            docs.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading txt files: {e}")
            
        try:
            docs.extend(md_loader.load())
        except Exception as e:
            print(f"Error loading md files: {e}")
            
        return docs
    else:
        raise ValueError(f"Invalid path: {path}")
