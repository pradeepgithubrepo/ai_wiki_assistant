from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents):
        split_docs = self.splitter.split_documents(documents)
        # Metadata (file_name, file_path) is automatically preserved in split_docs
        return split_docs
