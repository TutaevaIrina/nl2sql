# OpenAI API
import os
import json
from openai import OpenAI

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

# Define input and output file paths
input_file = "questions_experiment.txt"
output_file = "pred_example_enhanced_few_shot_prompt_2.txt"
base_dir = os.path.dirname(os.path.abspath(__file__))
tables_json_path = os.path.join(base_dir, "..", "..", "..", "spider_data", "spider_data", "tables.json")


def load_schema(database_name):
    """
    Loads the schema for the specified database from tables.json.
    """
    if not os.path.exists(tables_json_path):
        print("Error: tables.json file not found.")
        return None

    # Load the JSON data
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # Find the schema for the given database
    # https://github.com/BeachWang/DAIL-SQL/blob/main/prompt/PromptReprTemplate.py
    # https://github.com/BeachWang/DAIL-SQL/blob/main/generate_question.py
    
    for db in tables_data:
        if db["db_id"] == database_name:
            # Format schema metadata as a string for GPT-4o
            tables = db["table_names_original"]
            columns = db["column_names_original"]
            column_types = db["column_types"]
            foreign_keys = db["foreign_keys"]

            schema_metadata = []
            for table_idx, table_name in enumerate(tables):
                schema_metadata.append(f"Table: {table_name}")
                schema_columns = [
                    f"{col[1]} ({column_types[i]})"
                    for i, col in enumerate(columns)
                    if col[0] == table_idx
                ]
                schema_metadata.append(f"  Columns: {', '.join(schema_columns)}")

            if foreign_keys:
                schema_metadata.append("\nForeign Keys:")
                for fk in foreign_keys:
                    schema_metadata.append(
                        f"  {columns[fk[0]][1]} -> {columns[fk[1]][1]}"
                    )

            return "\n".join(schema_metadata)

    print(f"Schema not found for database: {database_name}")
    return None


def clean_output(raw_sql):
    """
    Cleans the raw SQL output to remove explanations, redundant text, and markers.
    """
    # Remove unnecessary text or explanations
    if "```sql" in raw_sql:
        raw_sql = raw_sql.split("```sql")[-1]  # Get content after "```sql"
    if "```" in raw_sql:
        raw_sql = raw_sql.split("```")[0]  # Remove everything after closing "```"

    # Remove explanations or introductory text
    lines = raw_sql.splitlines()
    cleaned_lines = [
        line
        for line in lines
        if not line.strip().startswith("To ") and not line.strip().startswith("Here")
    ]
    return "\n".join(cleaned_lines).strip()


def generate_sql(question, schema_metadata):
    """
    Generates SQL query for a given question and schema using GPT-4o.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
        "role": "system",
        "content": """You are an SQL expert. Your task is to generate correct SQL queries based on the user's questions and the provided database schema. Avoid unnecessary complexity in the queries. Focus on direct and concise solutions.
        Ensure all textual comparisons are case-insensitive using functions like LOWER().
        Avoid using LEFT JOIN, or CROSS JOIN unless explicitly required by the question or the database schema. Use INNER JOIN whenever possible for combining tables.
        Ensure the use of DISTINCT when the query may result in duplicate rows, especially in cases involving joins, unless duplicates are explicitly required by the question.
        Do not use LIMIT even when the question asks for the most, the least, the greatest, the highest or the smallest results. 
        

        Here are examples to guide your responses:

        ### CASE-INSENSITIVE EXAMPLES
        **Question:** How many car models are produced in the United States?
        SELECT COUNT(DISTINCT ml.Model) AS NumberOfCarModels FROM model_list ml JOIN car_makers cm ON ml.Maker = cm.Id JOIN countries c ON cm.Country = c.CountryId WHERE LOWER(c.CountryName) = LOWER('USA')

        ### JOIN EXAMPLES
        **Question:** How many concerts are performed at each stadium?
        SELECT s.Name AS Stadium_Name,	COUNT(c.concert_ID) AS Number_of_Concerts FROM stadium s JOIN concert c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID, s.Name;

        **Question:** List all template IDs along with the count of documents associated with each template.
        SELECT T.Template_ID, COUNT(D.Document_ID) AS NumberOfDocuments FROM Templates T JOIN Documents D ON T.Template_ID = D.Template_ID GROUP BY T.Template_ID;

        **Question:** List all possible combinations of breed types and size types.
        SELECT DISTINCT breed_code, size_code FROM dogs; 

        ### DISTINCT EXAMPLES
        **Question:** What are the different first names and ages of the students who do have pets?
        SELECT DISTINCT S.Fname, S.Age FROM Student S INNER JOIN Has_Pet HP ON S.StuID = HP.StuID;     

        ### GROUP BY EXAMPLES
        **Question:** List the ID, name, and age of visitors who have visited one or more museums more than once.
        SELECT t1.id, t1.name, t1.age FROM visitor AS t1 JOIN visit AS t2 ON t1.id = t2.visitor_id GROUP BY t1.id HAVING count(*) > 1

        **Question:** How many times at most can a course enrollment result show in different transcripts? Also, show the course enrollment id.
        SELECT student_course_id, COUNT(transcript_id) AS count FROM Transcript_Contents GROUP BY student_course_id HAVING COUNT(transcript_id) = (SELECT MAX(count) FROM (SELECT COUNT(transcript_id) AS count FROM Transcript_Contents GROUP BY student_course_id) AS SubQuery);   

        ### AGGREGATION EXAMPLES
        **Question:** What is the average age of dogs that received treatments?
        SELECT avg(age) FROM Dogs WHERE dog_id IN (SELECT dog_id FROM Treatments);  

        ### LOGICAL OPERATORS EXAMPLES
        **Question:** Find the average age of students who do not have any pet.
        SELECT AVG(Age) AS AvgAgeNoPets FROM Student WHERE StuID NOT IN (SELECT StuID FROM Has_Pet);

        **Question:** What is the series name and country of all TV channels that are playing cartoons directed by Ben Jones and cartoons directed by Michael Chang?
        SELECT DISTINCT TC.series_name, TC.Country FROM TV_Channel TC WHERE TC.id IN (SELECT c1.Channel FROM Cartoon c1 JOIN Cartoon c2 ON c1.Channel = c2.Channel WHERE c1.Directed_by = 'Ben Jones' AND c2.Directed_by = 'Michael Chang');
        
        **Question:** What are the names of properties that are either fields or shops with less than 20 million?
        SELECT property_name FROM Properties WHERE property_type_code = "Fields" OR (property_type_code = "Shop" AND agreed_selling_price < 20000000);  
        
        ### EXAMPLES TO AVOID LIMIT
        **Question:** What area code received the highest number of votes?
        SELECT T1.area_code, COUNT(*) FROM area_code_state AS T1 JOIN votes AS T2 ON T1.state = T2.state GROUP BY T1.area_code HAVING COUNT(*) = (SELECT MAX(vote_count) FROM (SELECT COUNT(*) AS vote_count FROM area_code_state AS S1 JOIN votes AS S2 ON S1.state = S2.state GROUP BY S1.area_code) AS SubQuery);
                
        **Question:** Which continent speaks the largest number of languages?
        SELECT c.Continent, COUNT(cl.Language) AS LanguageCount FROM country c JOIN countrylanguage cl ON c.Code = cl.CountryCode GROUP BY c.Continent HAVING COUNT(cl.Language) = (SELECT MAX(LanguageCount) FROM (SELECT COUNT(cl_sub.Language) AS LanguageCount FROM country c_sub JOIN countrylanguage cl_sub ON c_sub.Code = cl_sub.CountryCode GROUP BY c_sub.Continent) AS SubQuery);        

        ### EXAMPLES TO AVOID DISTINCT
        **Question:** Show the names and all grades of high schoolers.
        SELECT name, grade FROM Highschooler             
        """,
            },
            {
                "role": "user",
                "content": f"Given the following database schema:\n\n{schema_metadata}. Generate an SQLite SQL query only and with no explanation for the question {question}.",
            },
        ],
    )
    sql = completion.choices[0].message.content.strip()
    return clean_output(sql)


def main():
    # Read questions from the input file
    with open(input_file, "r", encoding="utf-8") as f:
        questions = f.readlines()

    # Open the output file to write the generated SQL queries
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, question in enumerate(questions, start=1):
            question = question.strip()
            if question:  # Skip empty lines
                # Extract database name from the question (assumes `|||` separates it)
                parts = question.split("|||")
                if len(parts) < 2:
                    print(f"Invalid format for question {idx}: {question}")
                    continue
                database_name = parts[1].strip()
                schema_metadata = load_schema(database_name)

                if not schema_metadata:
                    f.write(f"Question {idx}: {question}\n")
                    f.write("SQL: ERROR: Schema not found.\n\n")
                    continue

                print(f"Generating SQL for {question}")
                sql_query = generate_sql(parts[0].strip(), schema_metadata)
                f.write(f"{question}\n")
                f.write(f"SQL: {sql_query}\n\n")
    print(f"SQL queries saved to {output_file}")


if __name__ == "__main__":
    main()
