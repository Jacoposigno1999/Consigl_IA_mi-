from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import os

class SemanticReviewSearcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = "data/faiss_vector_DB"):
        self.index_path = index_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None  # will be initialized later, this enable other methods to reference it later

    def build_index(self, reviews: List[dict], content_field: str = "review_full"):
        """Embeds and indexes a list of review documents"""
        print(f"🔧 Building FAISS index with {len(reviews)} reviews...")
        
        docs = [
            Document(
                page_content=doc.get(content_field, ""),
                metadata={k: v for k, v in doc.items() if k != content_field}
            )
            for doc in reviews if doc.get(content_field)
        ]

        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.vectorstore.save_local(self.index_path)
        print(f"✅ Index saved at: {self.index_path}")

    def load_index(self):
        """Load a previously saved FAISS index"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index path '{self.index_path}' not found.")
        
        self.vectorstore = FAISS.load_local(self.index_path, embeddings=self.embedding_model)
        print(f"✅ Loaded FAISS index from {self.index_path}")

    def search(self, query: str, k: int = 3):
        """Run semantic search on the loaded vector DB"""
        if not self.vectorstore:
            raise RuntimeError("Vector index not loaded. Call load_index() first.")

        print(f"🔍 Semantic search for: {query}")
        results = self.vectorstore.similarity_search(query, k=k)
        return results #What type is this? I think List[Dict]
    
    def run_rag(self, retrived_reviews: List[dict], query: str, content_field: str = "review_full") -> str:
        """Perform a semantic RAG over the filtered reviews"""
        if not retrived_reviews:
            return "⛔ No reviews to perform RAG on."

        # Step 1: Convert reviews to LangChain Documents
        docs = [
            Document(page_content=doc[content_field], metadata=doc)
            for doc in retrived_reviews if content_field in doc
        ]

        if not docs:
            return "⛔ No valid review content to use in RAG."

        # Step 2: Build temporary FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Step 3: Define custom prompt
        prompt = PromptTemplate.from_template("""
        You are a food expert helping users find restaurants based on their preferences.

        Use ONLY the following restaurant reviews to answer the user's request.
        If the reviews are insufficient to answer, say: "I don’t have enough information."

        Context:
        {context}

        User query: {question}
        Answer:
        """)

        # Step 4: Build and run the RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        return rag_chain.run(query)
