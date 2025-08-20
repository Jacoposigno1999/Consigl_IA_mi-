import ast #for converting string into dict 
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from collections import Counter
from pymongo import MongoClient

class RestaurantRecommender:
    def __init__(self, db_name, collection_name, model_name):
      # LLM model setup
      self.model = OllamaLLM(model=model_name)

      # MongoDB setup
      try:
          self.conn = MongoClient("localhost", 27017)  #client = MongoClient("mongodb://localhost:27017")
          print("✅ MongoDB connected.")
      except Exception as e:
          raise ConnectionError(f"Could not connect to MongoDB: {e}")

      self.db = self.conn[db_name]
      self.collection = self.db[collection_name]
    
        
    def parse_query(self, user_input: str) -> dict:
      #==========================================================================
      # From user query to a dictionary with the information to look for in the DB
      #==========================================================================
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
      prompt = prompt_template.format(query = user_input)      
      output = self.model.invoke(prompt)
      return ast.literal_eval(output) #convert a string into a dict
      
      
    def build_mongo_query(self, parsed_info: Dict[str, List[str]]) -> Dict[str, Any]:
      
      # Merge all values from food_item, preferences, and aspects
      search_aspects = []
      for key in ["food_item", "preferences", "aspects"]:
        search_aspects.extend(parsed_info.get(key, []))
        
      # If nothing to search, return empty query (match all)
      if not search_aspects:
        return "⚠️ Be more specif, no aspect was found"
      
      query = {}
      # Create one regex matcher per term (case-insensitive, substring match)
      regex_filters = [
        {"aspect_keys": {"$elemMatch": {"$regex": term, "$options": "i"}}}
        for term in search_aspects] 
    
      query = {"$or": regex_filters}
      return query  

 
    def query_reviews(self, mongo_query: Dict[str, Any], print_info: bool = False) -> List[Dict[str, Any]]:
      
      # Run it
      results = list(self.collection.find(mongo_query))
      if print_info: 
        print(f"✅ Found {results.count_documents(mongo_query)} matching documents.")
        print("🔍 Showing up to 5 sample results:\n")
        for doc in results:
          print(f"🍽️ Restaurant: {doc.get('restaurant_name')}")
          print(f"📝 Aspects: {doc.get('aspect_keys')}")
          print()  
        
      return results
        

    def enrich_results(self, results: List[dict]) -> List[dict]:
      # =========================
      # Add other usefull metadata to the retrived informations 
      # For now just add a score to the information retrived
      # =========================
      #Counting how many times a resturant is present the query results
      resturant_apparence = Counter(doc['restaurant_name'] for doc in results) 

      enriched_results = [
        {**doc, 'review_count': [resturant_apparence[doc["restaurant_name"]]]} for doc in results 
        ]
      return enriched_results


    def get_top_restaurants(self, enriched_results: List[dict], top_k: int = 4) -> List[str]:
      # ====================
      # Select top resturants (the one with highest review_count and retrive all the information(all the reviews))
      # ====================

      enriched_results.sort(key = lambda doc: doc['review_count'], reverse= True )

      top_resturants = []
      for doc in enriched_results:
        if doc['restaurant_name'] not in top_resturants:
          top_resturants.append(doc["restaurant_name"])
        if len(top_resturants) > 4:
          return top_resturants;
        return top_resturants
        

    def fetch_full_reviews(self, top_restaurants: List[str]) -> List[Dict]:
        if not top_restaurants:
          print("⛔ No top resturants found")
          return []
        
        query = {"restaurant_name": {"$in": top_restaurants}}
        return list(self.collection.find(query))



















if __name__ == "__main__":


  request = input("Hello what you would like to eat?")


  


