from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
from Data_PreProcessing import ABSA_expert
import re


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#==========================================================================
# Thsi code is use to test the performance of simple LM in performing ABSA
# As 'silver standard" we will use a datased labeled by ChatGPT-o4 
#==========================================================================
path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\raw\\Barcelona_reviews.csv"
raw_data = pd.read_csv(path)

#====================================
# Creating a balanced dataset to be labeled

Neg = 250
Pos = 350


# Sample from each class
positive_sample = raw_data[raw_data['sample'] == 'Positive'].sample(n=Pos, random_state=42)
negative_sample = raw_data[raw_data['sample'] == 'Negative'].sample(n=Neg, random_state=42)

# Combine them into one balanced dataset
balanced_sample = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

benchmark_data = balanced_sample



#=================================================================
# Class used to produce a labeled dataset to be used as benchmark
#=================================================================

class Expert_Model_Labeler():
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.Absa_model = ChatOpenAI(
                                        model="gpt-4",
                                        temperature=0,
                                        openai_api_key=openai_api_key  # from .env
                                    ) 
        
        self.prompt_template = '''
            You are a Natural Language Processing (NLP) expert. Your task is to perform Aspect-Based Sentiment Analysis (ABSA) on the following food review.

                Identify all aspects mentioned or implied in the text (both explicit and implicit), and assign a sentiment polarity to each one. Aspects should refer to specific elements of the restaurant experience such as food, service, waiting time, prices, atmosphere, etc.
                
                Return the result as a dictionary formatted in JSON style:
                - Keys = aspect terms (e.g., "pizza", "service", "wait time")
                - Values = sentiment polarity: "positive", "negative", or "neutral"
                - Use lowercase for all aspect names
                - Only include aspects mentioned or implied in the review
                - Group synonymous or repeated terms under the same aspect (e.g., "wait time" instead of both "delay" and "waiting")
                
                ---
                
                Examples:
                
                Input:
                "Pizza was very good but the service was quite terrible. Prices were honest."
                
                Output:
                {{ "pizza": "positive", "service": "negative", "price": "positive" }}
                        
                Input:
                "We waited over an hour despite the promise of a 20-minute wait. Once seated, the pasta was undercooked and the waiter ignored us."
                
                Output:
                {{ "wait time": "negative", "pasta": "negative", "service": "negative" }}

                Input:
                "Empanadas were average but the outdoor patio was beautiful and relaxing. Fair prices too."

                Output:
                {{ "empanadas": "neutral", "atmosphere": "positive", "price": "positive" }}

                ---

                Now analyze the following review and return a JSON-style dictionary of {{aspect: sentiment}}:
                
                {food_review}
            
            '''
            
    def labeling(self, df: pd.DataFrame, 
                 save_path: str =  "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\processed\\labeled_data.csv", 
                 save_every: int = 50 ) -> pd.DataFrame: 
        
        labeled_df = df
        labeled_df["aspects"] = None
       
       
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        
        for idx, row in df.iterrows():
            
            review = row['review_full']
            
            if (idx +1) %10 == 0:
                print(f'Labeling progress: {idx}/{len(df)}')
                
            
            prompt = prompt_template.format(food_review = review)
            
            try:
                llm_response = self.Absa_model.invoke(prompt)
                content = llm_response.content

                json_str = re.search(r"{.*}", content, re.DOTALL).group(0)
                aspects = json.loads(json_str)
                
            except Exception as e:
                print(f"Error on index {idx}: {e}")
                aspects = {}
                
            labeled_df.at[idx, "aspects"] = aspects
        
            
            # Call the save method every N rows
            if (idx + 1) % save_every == 0:
                self.save(labeled_df.loc[:idx], save_path)
                
            #save whole df at the end     
        self.save(labeled_df, save_path) 
        return labeled_df 
    
    
    def save(self, df: pd.DataFrame, path: str):
        try:
            df.to_csv(path, index=False)
            print(f"[✔] Checkpoint saved to: {path}")
        except Exception as e:
            print(f"[✘] Failed to save checkpoint: {e}")
            
#==========================================================     
# Class to test different models on the same beanchmark df 
#==========================================================          
class Challenger_model():
    def __init__(self):
        pass


class Model_Eval():
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)


    def is_semantic_match(self, a1, a2, threshold: float ) -> bool:
        """Returns True if a1 and a2 are semantically similar above threshold."""
        emb1 = self.embedding_model.encode(a1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(a2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item() >= threshold
    
    
    # Evaluation function
    #with gpt_outputs we refers to the labeled dataset (which was labeled using ChatGPT-4)
    def evaluate_absa(self, gpt_outputs: List[Dict], model_outputs: List[Dict], threshold: float =0.5):
        results = []

        for idx, (gpt_out, model_out) in enumerate(zip(gpt_outputs, model_outputs)):
            gpt_items = list(gpt_out.items())
            model_items = list(model_out.items())

            matched = []
            sentiment_mismatches = []
            unmatched = []

            for m_aspect, m_sentiment in model_items:
                found_match = False
                for g_aspect, g_sentiment in gpt_items:
                    if self.is_semantic_match(m_aspect, g_aspect, threshold):
                        found_match = True
                        if  m_sentiment.lower() == g_sentiment.lower():
                            matched.append((m_aspect, m_sentiment))
                        else:
                            sentiment_mismatches.append((m_aspect, m_sentiment, g_sentiment))
                        break
                if not found_match:
                    unmatched.append((m_aspect, m_sentiment))

            precision = len(matched) / len(model_items) if model_items else 0
            recall = len(matched) / len(gpt_items) if gpt_items else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

            results.append({
                "review_index": idx,
                "matched": matched,
                "sentiment_mismatches": sentiment_mismatches,
                "unmatched": unmatched,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    
    Challenger  =  ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest")  


# Run the evaluation
df_evaluation = evaluate_absa(gpt_outputs, model_outputs)
import caas_jupyter_tools as cjtools; cjtools.display_dataframe_to_user(name="ABSA Evaluation Report", dataframe=df_evaluation)
