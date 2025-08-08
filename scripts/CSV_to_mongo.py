from pymongo import MongoClient
import pandas as pd

try:
    conn = MongoClient("localhost", 27017)
    print("Connected successfully!")
except Exception as e:
    print("Could not connect to MongoDB:", e)


path = "c:\\Users\\jacop\\Desktop\\Lavori\\Consigl_IA_mi-\\data\\processed\\absa_expanded_dataset.csv"
df = pd.read_csv(path)

df_dict = df.to_dict(orient="records")

db = conn["Reviews"]
db.Barcelona.insert_many(df_dict)




