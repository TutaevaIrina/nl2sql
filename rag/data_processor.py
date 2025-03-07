import os
import json
import sqlalchemy as sa
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings


embeddings_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

# Define file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
tables_json_path = os.path.join(base_dir, "..", "..", "spider_data", "spider_data", "tables.json")
database_folder = os.path.join(base_dir, "..", "..", "spider_data", "spider_data", "database")
faiss_index_path = "faiss_index.idx"
json_data_path = "faiss_text_data.json"

# List of databases
databases = [
    "concert_singer", "pets_1", "car_1", "flight_2", "employee_hire_evaluation",
    "cre_Doc_Template_Mgt", "course_teach", "museum_visit", "wta_1", "battle_death",
    "student_transcripts_tracking", "tvshow", "poker_player", "voter_1", "world_1",
    "orchestra", "network_1", "dog_kennels", "singer", "real_estate_properties"
]

# Load schema from the JSON file and format it as text
def load_schema(tables_json_path):
    with open(tables_json_path, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    schema_text = []
    for db in schema_data:
        db_id = db["db_id"]
        if db_id not in databases:
            continue
        tables = db["table_names_original"]
        columns = db["column_names_original"]

        schema = f"Database: {db_id}\n"
        for i, table in enumerate(tables):
            schema += f"Table: {table}\n"
            for col in columns:
                if col[0] == i:
                    schema += f"  - {col[1]}\n"

        schema_text.append(schema)

    return schema_text

# Convert schema data into vectors and store them in FAISS
def store_schema_as_vectors(schema_data):    

    print("Creating embeddings...")
    vector_data = embeddings_model.embed_documents(schema_data)
    
    dimension = len(vector_data[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vector_data))

    # FAISS-Index lokal speichern
    faiss.write_index(index, "faiss_index.idx")
    print("FAISS-Index successfully saved.")

    # Save original text in JSON file
    with open(json_data_path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=4)
    
    print("Schema saved in JSON file")
    
    return index


# Load data
if __name__ == "__main__":
    schema_texts = load_schema(tables_json_path)    

    # FAISS- and JSON-Storing
    vectorstore = store_schema_as_vectors(schema_texts)

    print("Schema has been successfully stored in FAISS and JSON file")
