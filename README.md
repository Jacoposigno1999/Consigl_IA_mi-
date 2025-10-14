# Building a Smart Restaurant Recommender Agent leveraging Aspect-Based Sentiment 

The project was developed with the  goal of creating a smart restaurant recommender agent capable of tailoring its suggestions to the specific preferences and expectations of users. Unlike traditional systems that rely on average ratings or general review summaries, the agent system should understand what kind experience (e.g., food, service, atmosphere) the user is looking for, and suggest the restaurant that best matches its requests.

To achieve this, a key requirement emerged: the ability to perform **Aspect-Based Sentiment Analysis (ABSA)** on restaurant reviews. ABSA makes it possible to break down a review into **specific aspects**, each linked to a **sentiment polarity**, enabling a more nuanced understanding of user feedback and consequently the ability to provide more accurate suggestions.

## 2. Methodology

The development of the system followed a structured methodology composed of several components.

The choice of the various components and their implementation was dictated by several objectives and constraints:

- Obtain excellent results in a reasonable amount of time 
- Can be executed locally (taking into account all computational limitations)
- Development of robust code (which could easily be put into production without the need for extensive modifications) 

The idea was therefore to create a project more similar to a corporate project than a research project.
All this represented a major challenge in terms of the trade-off between output quality and precision and computational resource expenditure.

<img width="800" height="338" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/31d4cfff-89e8-4b6b-bd8d-7a6afd17526f" />


### 2.1. Data Acquisition and ABSA Preprocessing pipeline

Restaurant reviews were obtained from a raw dataset (in this case, focused on Barcelona) and loaded into a Pandas DataFrame. 

After some basic cleaning operations were applied, a pre-trained, instruction-tuned model from HugginFace (`pyabsa-v3-onlyRest`) was used to process each review and extract relevant **aspect–sentiment pairs**.

Each review was processed individually using a prompt-driven generation approach. The model’s output — a dictionary with aspect terms as keys and sentiment polarities as values — was parsed and stored in two new columns:

- `aspects`: A dictionary in the form `{aspect: sentiment}`
- `aspect_keys`: A list of extracted aspect terms (keys only), useful for later filtering or retrieval.

This transformation effectively **augmented the original dataset** with structured semantic insights, preparing it for downstream recommendation logic and evaluation. The augmented dataset is then saved in a  **MongoDB**[^1] .


Example of a document saved in MongoDB. 
```
{
  "_id": {
    "$oid": "68a32681150e99c4db0b6138"
  },
  "restaurant_name": "Sports_Bar_Italian_Food_C_Ample",
  "rating_review": 5,
  "sample": "Positive",
  "review_id": "review_750611106",
  "title_review": "Great vibe, great food",
  "review_full": "Visited here twice on my last trip to the city. Recommended to me and I'd  recommend to anyone who likes good Italian food. It's loud and bustling so not ideal for a romantic dinner. The pizzas are sublime, always get the foccacia, and try both the vegetable and meat lasagne, they're superb.",
  "date": "March 12, 2020",
  "city": "Barcelona_Catalonia",
  "url_restaurant": "https://www.tripadvisor.com/Restaurant_Review-g187497-d3379888-Reviews-Sports_Bar_Italian_Food_C_Ample-Barcelona_Catalonia.html",
  "author_id": "UID_15",
  "aspects": {
    "pizzas": "positive",
    "foccacia": "positive",
    "vegetable and meat lasagne": "positive"
  },
  "aspect_keys": [
    "pizzas",
    "foccacia",
    "vegetable and meat lasagne"
  ]
}
```

### 2.3. User Query Interpretation

Once the dataset was enriched and stored in MongoDB, the system became ready to support user interactions. The goal at this stage was to interpret natural language queries from users — such as _“I’d like some authentic pasta in a cozy place with good service”_ — and return smart, personalized restaurant recommendations based on semantic understanding of the reviews.


> [!DANGER] Computational  Limitation
> 
*Performing a semantic search on the entire dataset would have been too costly, as it would have meant embedding and indexing all the reviews in the database.*

To overcome computational limitations, the system uses a lightweight language model (`llama3:8b`) to **semantically parse the user's query** and extract the 'aspects keywords' related to the user request :

es. _“I’d like some authentic pasta in a cozy place with good service”_ --> `['pasta', 'cozy place', 'service']`

### 2.4. Top restaurant reviews retrieval pipeline

The extracted keywords are then used to **build a dynamic MongoDB query** targeting the `aspect_keys` field of each review document. This means that the system retrieves only reviews that mention at least one of the aspects the user is interested in[^5]. This allows us to greatly reduce the number of reviews to consider in order to recommend the restaurant best suited to the customer's request.

With this subset of relevant reviews, the system performs an internal aggregation by counting how many times each restaurant appears in the results. This count serves as a **proxy score for relevance** — restaurants that appear more frequently (with a positive sentiment) are assumed to be the most relevant making them more likely to match the user's intent. [^2]

The top 4 scoring restaurants are then selected and their names saved. 

### 2.5. Retrieval-Augmented Generation (RAG)

After identifying the top 4 restaurants that best match the user’s request, **all available reviews for these restaurants are retrieved from MongoDB**, and  and subsequently embedded and indexed into a temporary in-memory FAISS vector database.

Rather than relying on semantic search to select a subset of reviews, the system uses a "soft" form of Retrieval-Augmented Generation (RAG): **all the reviews saved in FAISS are retrived  and  passed as context** to a large language model  (`llama3:8b`)  via a custom prompt. This ensures that the language model has access to the full range of user opinions about the top restaurants.[^3]

The prompt explicitly instructs the model to:

- Recommend the top restaurant and justify why
    
- Compare it to the three alternatives
    
- Base the response **strictly** on the content of the provided reviews

## Results and Model Evaluation

The system works and provides recommendations that are consistent with what the user asks for and that are effectively reflected in what is written in the reviews.

However, it is difficult to assess whether the advice given by the agent is actually the most desirable.[^4]

It is clear that the quality of the final output depends largely on the ability of the model responsible for performing ABSA, as the entire retrieval process starts with identifying reviews that contain certain aspects requested by the user.

For this reason, a script  that allows us to evaluate the quality of ABSA results has been written(`ABSA_model_eval.py`, look at Appendix A for more information).



[^1]: MongoDB was chosen to support flexible and efficient querying, allowing the system to retrieve only the reviews relevant to a user query or a specific set of aspects.
	

[^2]: You might object that in this way that little-known restaurants will never be recommended over restaurants with many reviews (both negative and positive). This can be avoided by using very specific search terms that will only match  few specific reviews, allowing you to find little-known restaurants with very accurate clients reviews. However, this remains a problem that needs to be better understood and resolved. 

[^3]: Although the approach does not involve querying the vector store via semantic similarity, this step is still considered a simplified or **"soft" RAG**, as it augments the generation step with dynamically retrieved, grounded context. 

[^4]: Especially considering that for now it has only been tested in Barcelona's restaurants, which are completely unknown to me.

[^5]: This part definitely needs improvement. One possible improvement would be to first filter the reviews that match all the aspects extracted from the query, and only if there are not enough of them, move on to filtering for reviews that contain even just one of the extracted aspects. 

