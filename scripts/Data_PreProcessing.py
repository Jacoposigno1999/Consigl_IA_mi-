#================================
# This script take as input a Dataframe containing resturants reviews
# and augment it with 2 columns [''] containg {aspect:polarity} dictionaries
# and [''] containing a list with the aspects key
#================================
import  pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from typing import List, Dict
from pymongo import MongoClient



class ABSA_expert:
    def __init__(self, tokenizer: str, model: str , device: str = None ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
        self.model.eval() 
        
         # Prompt template for instruction-tuned models
        self.bos_instruction =  """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu:positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food:positive, menu:positive, service:positive, setting:positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
        output: toast:negative, mayonnaise:negative, bacon:negative, ingredients:negative, plate:negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
        output: seats:negative
        Neutral example 1-
        input: I asked for seltzer with lime, no ice.
        output: seltzer with lime:neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another.
        output: glass of wine:neutral
        Now complete the following example-
        input:"""
        self.delim_instruct = ''
        self.eos_instruction = ' \noutput:'
        
    def _build_prompt(self, review: str) -> str:
        return f"{self.bos_instruction}{review}{self.delim_instruct}{self.eos_instruction}" #probabilment delim_instruct non è necessario 
    
    """ Convert model string output to a dict of {aspect: sentiment}"""
    def _parse_output(self, output: str) -> Dict[str, str]:
        result = {}
        for item in output.split(","):
            parts = item.strip().split(":")
            if len(parts) == 2: #checking if we have aspect:sentiment format
                aspect, sentiment = parts
                result[aspect.strip()] = sentiment.strip()
        return result
        
    def analyze_review(self, review: str) -> Dict[str, str]:
        prompt = self._build_prompt(review)
        inputs = self.tokenizer(prompt, return_tensors="pt")#convert a string prompt into numerical tensors that the model can understand.
        inputs = {k: v.to(self.device) for k, v in inputs.items()} #Moves them to GPU if available (or CPU otherwise).

        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"])# se da errore prova: inputs.input_ids
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)#decoded output
        return self._parse_output(decoded)
    
    #PER ORA I DATI ARRIVANO DA EXCELL QUINDI LI CARICO COME UN PANDAS DF, SUCCESSIVAMENTE ARRIVERANNO DA UN MONGO-DB
    def analyze_dataset(self, df: pd.DataFrame, text_column: str = "review_full",
                        mongo_uri: str =  "mongodb://localhost:27017", db_name: str = "Reviews",
                        collection_name: str = "Barcelona", batch_size: int = 1000,
                        upload_to_mongo: bool = True) -> pd.DataFrame:
        
        if upload_to_mongo: 
            client = MongoClient(mongo_uri)
            collection = client[db_name][collection_name]
        
        all_aspects = []
        buffer = [] #buffer of observations to be loaded in batch to MongoDB

        for idx, row in df.iterrows():
            review = row[text_column]
            aspects = self.analyze_review(review)
            row_dict = row.to_dict() #convert everything to a dict, since we want a json document to be loaded on MongoDB
            row_dict["aspects"] = aspects
            row_dict["aspect_keys"] = list(aspects.keys())
            buffer.append(row_dict)
            all_aspects.append(aspects)

            if idx % 50 == 0:
                print(f"Processed {idx+1}/{len(df)} reviews...")
                
            if (idx + 1) % batch_size == 0 and upload_to_mongo:
                try: 
                    collection.insert_many(buffer)
                    print(f"✅ Uploaded batch of {len(buffer)} reviews at index {idx + 1}")
                except Exception as e:
                    print(f"⚠️ Failed to upload batch: {e}")
                buffer = []
                
        # Upload remaining items
        if buffer and upload_to_mongo:
            try:
                collection.insert_many(buffer)
                print(f"✅ Uploaded final batch of {len(buffer)} reviews.")
            except Exception as e:
                print(f"⚠️ Failed to upload batch: {e}")

        df["aspects"] = all_aspects  
        return df

''' 
#Sanity check 
analyzer = ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest")
print(analyzer.analyze_review("The pizza was amazing and the waiter was rude."))       
'''

if __name__ == "__main__":
    analyzer = ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest")  
    path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\raw\\Barcelona_reviews.csv"
    raw_data = pd.read_csv(path)
    #Removing useless columns 
    raw_data.drop('Unnamed: 0', axis = 1)
    test_data = raw_data.head(10)
    df_with_aspects = analyzer.analyze_dataset(test_data)





