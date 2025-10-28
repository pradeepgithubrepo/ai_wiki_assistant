from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def rerank(self, query: str, documents: list, top_k: int = 5):
        """
        Rerank retrieved documents using a cross-encoder model.
        """
        pairs = [(query, doc.page_content) for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc.rerank_score = float(score.item())

        # Sort by score (descending)
        ranked = sorted(documents, key=lambda x: x.rerank_score, reverse=True)

        return ranked[:top_k]
