from langchain.vectorstores import FAISS
from langchain.schema import Document
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM


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
        print(f"✅ In-memory FAISS index built with {len(docs)} documents.")
        #self.vectorstore.save_local(self.index_path)
        #print(f"✅ Index saved at: {self.index_path}")

    "Not needed if you don't want to keep the vector DB in memory (I think)"
    def load_index(self):
        """Load a previously saved FAISS index"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index path '{self.index_path}' not found.")
        
        self.vectorstore = FAISS.load_local(self.index_path, embeddings=self.embedding_model)
        print(f"✅ Loaded FAISS index from {self.index_path}")

    "Not needed since we want to retrive all the reviews in the Vector DB"
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Run semantic search on the loaded vector DB"""
        if not self.vectorstore:
            raise RuntimeError("Vector index not loaded. Call load_index() first.")

        print(f"🔍 Semantic search for: {query}")
        results = self.vectorstore.similarity_search(query, k=k)
        return results 
    
    def run_rag(self, top_restaurants: List[str], retrived_reviews: List[dict], query: str, content_field: str = "review_full", model: str = "llama3:8b") -> str:
        #================
        # This methon run a RAG on all the embeded reviews 
        # previously filtered based on aspects_keys
        #================
        """Perform a semantic RAG over the filtered reviews"""
        if not retrived_reviews:
            return "⛔ No reviews to perform RAG on."

        if not self.vectorstore:
            return "⛔ No valid review content to use in RAG."

        # 1. Retrieve ALL embedded docs
        docs = list(self.vectorstore.docstore._dict.values())
        # 2. Build the context
        max_chars = 100000
        context = "\n\n".join([doc.page_content for doc in docs])
        context = context[:max_chars]

        # Step 3: Define custom prompt
        
        prompt_template  = PromptTemplate.from_template("""
            You are a food expert helping users find the best restaurants based on detailed customer reviews.

            Use ONLY the following restaurant reviews as context for your answer.

            Recommend **{first_option}** as the top choice. Clearly explain why this restaurant stands out using the reviews provided.

            Then, suggest these other valid alternatives: **{alternatives}**.
            Briefly compare them to the first option, highlighting what is better or worse based on the context.

            If the reviews are insufficient to answer, say: "I don’t have enough information."

            Context:
            {context}

            User query: {question}

            Answer:
            """)

        # Step 4: Format the prompt
        prompt = prompt_template.format(first_option = top_restaurants[0], alternatives = top_restaurants[1:], context=context, question=query)

        llm = OllamaLLM(model=model)
        
        # Step 5: Get answer from the LLM
        response = llm.invoke(prompt)

        return response

























