import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def __init__(self, directory: str, max_workers: int = 4):
        self.directory = directory
        self.max_workers = max_workers

    def _load_single_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        print(f"Loaded {file_path}")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_path"] = file_path
        return docs

    def _get_all_pdf_files(self):
        """Recursively collect all PDF files under the directory"""
        pdf_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def load_pdfs(self):
        print(f"Loading PDFs recursively from: {self.directory}")
        documents = []
        pdf_files = self._get_all_pdf_files()
        print(f"Found {len(pdf_files)} PDF files.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_single_pdf, f): f for f in pdf_files}
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {futures[future]}: {e}")

        print(f"Total documents loaded: {len(documents)}")
        return documents
