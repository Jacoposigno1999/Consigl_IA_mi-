import  pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

#==
#Loading data
#==
path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\raw\\Barcelona_reviews.csv"
raw_data = pd.read_csv(path)

row = raw_data.iloc[24960]['review_full']



PROMPT_TEMPLATE = '''
You are a NLP expert, your objective is to perform Aspect Base Sentiment Anlysis on this food review :
{food_review}.

Return table with sub-string of the original review, each with the corresponding topic and associated sentiment.

The returned table should be should be a string with this formatting:  "Sentence | Aspect | Sentiment \\nThe pizza was great | pizza | positive\\n..."
Attention: be sure to put the \n at the end of each row 
'''

####### Testing if it works #######
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(food_review = row)

model = OllamaLLM(model="mistral")

response_text = model.invoke(prompt)
print(response_text)
#####################################


# Function to simulate LLM ABSA output parsing
def parse_llm_output(llm_response_str):
    # Example LLM response:
    # "Sentence | Aspect | Sentiment\nThe pizza was great | pizza | positive\n..."
    rows = []
    for line in llm_response_str.strip().split("\\n"):
        #print(f"line: {line}")
        if "Aspect" in line or "Sentence" in line:
            continue  # used to skip header
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            rows.append({
                "aspect_sentence": parts[0],
                "aspect": parts[1],
                "sentiment": parts[2]
            })
    return rows

def absa(df):
    # New dataset
    absa_rows = []

    for idx, row in df.iterrows():
        original_review = row['review_full']
        #print(f'ORIGINAL REVIEW : {original_review}')
        
        if idx%10 == 0:
            print(f'Progress: {idx}/{len(df)}')
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(food_review = original_review)
        
        # 1. Call your LLM with the review (this is a placeholder)
        llm_response = model.invoke(prompt)  
        #print(F'LLM RESPONSE: {llm_response}') 

        # 2. Parse the output into structured triples
        triples = parse_llm_output(llm_response)
        #print(f'print triples: {triples}')

        # 3. For each triple, copy metadata and append
        for triple in triples:
            absa_rows.append({
                **triple,
                "restaurant_name": row['restaurant_name'],
                "review_id": row['review_id'],
                "rating_review": row['rating_review'],
                "city": row['city'],
                "date": row['date'],
                "original_review": original_review,
               
            })
        #print('##################  Added new review to absa_df  ############################')

    # Final DataFrame
    absa_df = pd.DataFrame(absa_rows)
    absa_df.to_csv("c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\processed\\absa_expanded_dataset.csv", index=False)
    return absa_df



test_data = raw_data.head(5000)



if __name__ == "__main__":
    df = absa(test_data)


#import openpyxl
#absa_df.to_excel("absa_expanded_dataset.xlsx", index=False)



