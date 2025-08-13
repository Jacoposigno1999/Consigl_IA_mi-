import  pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from typing import List, Dict

#==
#Loading data
#==
path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\raw\\Barcelona_reviews.csv"
raw_data = pd.read_csv(path)

row = raw_data.iloc[24960]['review_full']


class ABSA_expert:
    def __init__(self, tokenizer: str, model: str , device: str = None ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
        self.model.eval() 
        
         # Prompt template for instruction-tuned models
        self.bos_instruction =  """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                                    Positive example 1-
                                    input: I charge it at night and skip taking the cord with me because of the good battery life.
                                    output: battery life:positive, 
                                    Positive example 2-
                                    input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                                    output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
                                    Negative example 1-
                                    input: Speaking of the browser, it too has problems.
                                    output: browser:negative
                                    Negative example 2-
                                    input: The keyboard is too slick.
                                    output: keyboard:negative
                                    Neutral example 1-
                                    input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
                                    output: battery:neutral
                                    Neutral example 2-
                                    input: Nightly my computer defrags itself and runs a virus scan.
                                    output: virus scan:neutral
                                    Now complete the following example-
                                    input: """
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
        inputs = self.tokenizer(prompt, return_tensors="pt")#.to(self.device)#tokenized_test
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"])# se da errore prova: inputs.input_ids
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)#decoded output
        return self._parse_output(decoded)
    
    #PER ORA I DATI ARRIVANO DA EXCELL QUINDI LI CARICO COME UN PANDAS DF, SUCCESSIVAMENTE ARRIVERANNO DA UN MONGO-DB
    def analyze_dataset(self, df: pd.DataFrame, text_column: str = "review_full") -> pd.DataFrame:
        all_aspects = []

        for idx, row in df.iterrows():
            review = row[text_column]
            aspects = self.analyze_review(review)
            all_aspects.append(aspects)

            if idx % 50 == 0:
                print(f"Processed {idx+1}/{len(df)} reviews...")

        df["aspects"] = all_aspects  
        return df

    
#Sanity check 
analyzer = ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest")
print(analyzer.analyze_review("The pizza was amazing and the waiter was rude."))       


if __name__ == "__main__":
    analyzer = ABSA_expert("Iceland/pyabsa-v3-onlyRest", "Iceland/pyabsa-v3-onlyRest")  
    path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\raw\\Barcelona_reviews.csv"
    raw_data = pd.read_csv(path)
    test_data = raw_data.head(100)
    df_with_aspects = analyzer.analyze_dataset(test_data)

print(df_with_aspects.head())




