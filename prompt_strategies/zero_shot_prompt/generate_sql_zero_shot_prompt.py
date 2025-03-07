# OpenAI API
import os
import json
from openai import OpenAI

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

# Define input and output file paths
input_file = "questions_experiment.txt"
output_file = "pred_example_zero_shot_prompt_2.txt"
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
                "content": """
                You are an SQL expert. Your task is to generate correct SQL queries based on the user's questions and the provided database schema. Avoid unnecessary complexity in the queries. Ensure that the queries are correct, concise, and follow these rules:

                1. **Case Sensitivity**:
                    - Use functions like `LOWER()` for case-insensitive comparisons if the database schema does not guarantee consistent case.
                
                2. **Literal Values**:
                    - In the database, the column `CountryName` stores values like 'usa', not 'United States'. 

                3. **Joins**:
                    - Use `INNER JOIN` by default for combining tables, unless the schema or the question explicitly requires `LEFT JOIN` or `CROSS JOIN`.
                    
                4. **DISTINCT Usage**:
                    - Always use `DISTINCT` if the question implies unique results and there is potential for duplicates due to joins or data structure.    

                5. **GROUP BY Rules**:
                    - Only include columns in the `GROUP BY` clause that uniquely identify each group.
                    - Avoid grouping by non-unique attributes such as `name` or `description` unless explicitly required.                

                6. **Aggregation Rules**:
                    - When using aggregation functions like `SUM`, `AVG`, `COUNT`, ensure that the grouping and filtering logic align with the question.
                    - Avoid overcounting or double-counting records. For example:
                        - Use subqueries or `DISTINCT` to ensure uniqueness in aggregation where needed.
                        - For averages, ensure that each entity is counted only once (e.g., one dog, regardless of multiple treatments).

                7. **Logical Operators**:
                    - Use `EXCEPT` or `NOT IN` for set-based exclusions rather than `LEFT JOIN` with `IS NULL`, unless explicitly required.
                
                8. **LIMIT Usage**:
                    - Do not use `LIMIT` to filter results for questions asking for the "most," "least," "greatest," or "smallest."
 
                Here are some rules specific to common scenarios:
                - For questions asking for "total," "average," or "sum," ensure that the calculation is scoped correctly.
                - For exclusion queries (e.g., "not used by any document"), avoid including partial matches due to NULL values in `LEFT JOIN`.
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
