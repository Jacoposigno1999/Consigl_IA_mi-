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
import ast


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#==========================================================================
# This script is use to test the performance of simple LM in performing ABSA
# As "silver standard" we will use a datased labeled by ChatGPT-o4 
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
class Model_Eval():
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)


    def is_semantic_match(self, model_list: str, labeled_list: str, threshold: float ) -> bool:
        """Returns True if a1 and a2 are semantically similar above threshold."""
        emb1 = self.embedding_model.encode(model_list, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(labeled_list, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item() >= threshold
    
    
    # Evaluation function
    #with gpt_outputs we refers to the labeled dataset (which was labeled using ChatGPT-4)
    def evaluate_absa(self, gpt_outputs: List[Dict], model_outputs: List[Dict], threshold: float =0.5):
        results = []

        for idx, (gpt_out, model_out) in enumerate(zip(gpt_outputs, model_outputs)):
            #Dictionary Level 
            gpt_items = list(gpt_out.items())
            model_items = list(model_out.items())

            matched = []
            sentiment_mismatches = []
            unmatched = []


            for g_aspect, g_sentiment in gpt_items:
                #Key : Value Level 
                found_match = False
                for m_aspect, m_sentiment in model_items:
                    if self.is_semantic_match(g_aspect, m_aspect, threshold):
                        found_match = True
                        if  m_sentiment.lower() == g_sentiment.lower():
                            matched.append((m_aspect, m_sentiment))
                        else:
                            sentiment_mismatches.append((m_aspect, m_sentiment, g_sentiment))
                        break
                if  found_match == False:
                    unmatched.append((g_aspect, g_sentiment))


            precision = len(matched) / len(model_items) if model_items else 0
            recall = len(matched) / len(gpt_items) if gpt_items else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

            results.append({
                "review_index": idx,
                "Expert_Label": gpt_items,
                "Model_Prediction":  model_items,
                "matched": matched,
                "unmatched": unmatched,
                "sentiment_mismatches": sentiment_mismatches,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        results_df = pd.DataFrame(results)
        return results_df


    def save_evaluation_report(self, report_path: str, results_df: pd.DataFrame) -> None:
        
        avg_precision = results_df["precision"].mean()
        avg_recall = results_df["recall"].mean()
        avg_f1 = results_df["f1_score"].mean()
        summary_df = pd.DataFrame([{
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1_score": avg_f1
            }])
        
        
        print(f"\n📊 Average Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1: {avg_f1:.3f}")
        
        
        with pd.ExcelWriter(report_path, engine='openpyxl', mode='w') as writer:
            results_df.to_excel(writer, sheet_name="Evaluation", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
        print(f"✅ Report successfully saved to: {report_path}")

    
       
#=======================
#Functiond that trasform a df column containing dictionary to a list of dict
#=======================
def col_to_list(df: pd.DataFrame, col_name: str) -> List[Dict]:
    return df[col_name].tolist()
    
    
if __name__ == "__main__":
    
    Challenger  =  ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest") 
    Challenger_df =  Challenger.analyze_dataset(benchmark_data, upload_to_mongo = False) 
    model_name = "Iceland_pyabsa-v3-onlyRest"
    path = f'c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\processed\\{model_name}_challenger_data.csv'
    Challenger_df.to_csv(path, index=False)
    
    Challenger_list = col_to_list(Challenger_df, 'aspects')

    Labeled_df = pd.read_csv("c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\processed\\labeled_data.csv")
    Labeled_df["aspects"] = Labeled_df["aspects"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})#needed otherwise it upload 'aspects' as strings
    Labeled_list = col_to_list(Labeled_df, 'aspects')
    
    Evaluator = Model_Eval()
    Eval_results = Evaluator.evaluate_absa(Labeled_list, Challenger_list )
    
    report_path = f"c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\reports\\Models_eval\\{model_name}_eval_results.xlsx"
    
    Evaluator.save_evaluation_report(report_path,Eval_results )
    
