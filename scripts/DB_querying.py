
#================
# This script translate the user query into ad MongoDB query, in order to 
# retrive information from the MongoDB database
#================ 
import ast #for converting string into dict 
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

#==========================================================================
# From user query to a dictionary with the information to look for in the DB
#==========================================================================
request = input("Hello what you would like to eat?")


PROMPT_TEMPLATE = '''
You are a NLP expert, your objective is to extract the main food item, special preferences, 
and sentiment criteria from the query: 
{query}.

<expected output>
{{
  "food_item": [],
  "preferences": [],
  "aspects": []
}}
<\expected output>

Notice: 
- If some of the fields are not explicitly mentioned, don't include them in the dictionary 
- Do non return anything else apart what explicitelly asked in <expected output>, 
  do not print something like 'Base on the query..." or other notes
'''
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(query = request)

model = OllamaLLM(model="llama3:8b")
info = model.invoke(prompt)
info = ast.literal_eval(info)

print(info)


#=======================
# Query the database based on the infos
#=======================
from pymongo import MongoClient

try:
    conn = MongoClient("localhost", 27017)
    print("Connected successfully!")
except Exception as e:
    print("Could not connect to MongoDB:", e)
    
db = conn["Reviews"]
collection = db["Barcelona"] 

# Build the query
mongo_query = {
    "aspect": {"$in": info['food_item']}
}

# Execute the query
results = list(collection.find(mongo_query))

    
# =========================
# Add other usefull metadata to the retrived informations 
# For now just add a score to the information retrived
# =========================
from collections import Counter

#Counting how many times a resturant is present the query results
resturant_apparence = Counter(doc['restaurant_name'] for doc in results) 

enriched_results = [
  {**doc, 'review_count': [resturant_apparence[doc["restaurant_name"]]]} for doc in results 
]


# ====================
# Select top resturants (the what with highest review_count and retrive all the information(all the reviews))
# ====================

enriched_results.sort(key = lambda doc: doc['review_count'], reverse= True )

top_resturants = []
for doc in enriched_results:
  if doc['restaurant_name'] not in top_resturants:
    top_resturants.append(doc["restaurant_name"])
  if len(top_resturants) > 4:
    break;
  

query = {
  "resturant_name":{"$in": top_resturants}
}
    
top_rest_info = collection.find(mongo_query)

unique_review_id = top_rest_info.distinct('review_id')

query = {
  "resturant_name":{"$in": top_resturants},
   "review_id" : {"$in": unique_review_id }
}
top_rest_info = list(collection.find(mongo_query))




