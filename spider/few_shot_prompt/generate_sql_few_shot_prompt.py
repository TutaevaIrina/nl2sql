# OpenAI API
import os
import json
from openai import OpenAI

client = OpenAI(api_key = os.environ.get("OPEN_API_KEY"))

# Define input and output file paths
input_file = "questions_experiment.txt"
output_file = "pred_example_few_shot_prompt_2.txt"
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
        "content": """You are an SQL expert. Your task is to generate correct SQL queries based on the user's questions and the provided database schema.

        Here are examples to guide your responses:        

        ### DISTINCT EXAMPLES
        **Question:** Find the first name and age of each pupil who has a brother.
        SELECT DISTINCT p.pupil_id, p.first_name, p.age 
        FROM pupils AS p 
        JOIN has_siblings AS hs ON p.pupil_id = hs.pupil_id 
        WHERE LOWER(hs.relationship) = LOWER('Brother');

        ### AGGREGATION AND GROUP BY EXAMPLES
        **Question:** What are the product names and total sales for products in the "Electronics" category that have sold more than 500 times?
        SELECT p.product_name, SUM(o.quantity) AS total_sold
        FROM Products p
        JOIN Orders o ON p.product_id = o.product_id
        WHERE LOWER(p.category) = LOWER('Electronics')
        GROUP BY p.product_id
        HAVING SUM(o.quantity) > 500;

        **Question:** What is the average salary of employees who have completed at least one training program?
        SELECT AVG(e.salary) AS average_salary
        FROM Employees e
        WHERE e.employee_id IN (
                SELECT t.employee_id
                FROM Trainings t
            );     

        ### LOGICAL OPERATORS EXAMPLES
        **Question:** Show all department names in the United States that do not have any employees.
        SELECT department_name 
        FROM Departments 
        WHERE LOWER(country) = LOWER('USA') 
        EXCEPT 
        SELECT d.department_name 
        FROM Departments d 
        JOIN Employees e 
        ON d.department_id = e.department_id
        WHERE LOWER(d.country) = LOWER('USA');

        **Question:** Find the book title and author of the books written by both J.K. Rowling and Neil Gaiman.
        SELECT B1.title, B1.author
        FROM Books AS B1
        JOIN Authors AS A1 ON B1.author_id = A1.author_id
        WHERE LOWER(A1.name) = LOWER('J.K. Rowling')
        INTERSECT
        SELECT B2.title, B2.author
        FROM Books AS B2
        JOIN Authors AS A2 ON B2.author_id = A2.author_id
        WHERE LOWER(A2.name) = LOWER('Neil Gaiman');
        
        **Question:** What are the names of vehicles that are either SUVs or sedans with an engine size greater than 2.0 liters?
        SELECT vehicle_name
        FROM Vehicles
        WHERE LOWER(vehicle_type) = LOWER('SUV')
        UNION
        SELECT vehicle_name
        FROM Vehicles
        WHERE LOWER(vehicle_type) = LOWER('Sedan') AND engine_size > 2.0;         
                
        ### EXAMPLES TO REDUCE COMPLEXITY              
        **Question:** Find the name and total sales of all authors who sold the most books.
        WITH AuthorSales AS (
            SELECT author_name, SUM(sales) AS total_sales
            FROM books
            GROUP BY author_name),
        MaxSales AS (
            SELECT MAX(total_sales) AS max_sales
            FROM AuthorSales
            )
        SELECT author_name, total_sales
        FROM AuthorSales
        WHERE total_sales = (SELECT max_sales FROM MaxSales);        

        ### EXAMPLES TO AVOID MISUNDERSTANDING
        **Question:** What are the membership options and the class names available in fitness centers that offer swimming pools?
        SELECT membership_option, class_name
        FROM Fitness_Centers
        WHERE LOWER(has_swimming_pool) = LOWER("yes");

        **Question:** What are the categories where the highest percentage of products are on discount?
        SELECT c.category_name, p.discount_percentage
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
        WHERE p.discount_percentage = (
            SELECT MAX(sub_p.discount_percentage)
            FROM products sub_p
            WHERE sub_p.category_id = p.category_id
          )
        ORDER BY p.discount_percentage DESC;

        **Question:** Show all event dates in the sports calendar
        SELECT event_date, COUNT(*) AS event_count
        FROM Sports_Events
        GROUP BY event_date
        ORDER BY event_count DESC;

        ### EXAMPLES TO AVOID CROSS JOIN
        **Question:** What are all the possible department and job title combinations?
        SELECT DISTINCT department_name, job_title
        FROM Employees;

        ### EXAMPLES TO AVOID LIMIT
        **Question:** Which course has the most number of enrollments?
        SELECT c.course_name, COUNT(*) AS NumberOfEnrollments
        FROM enrollments e
        JOIN courses c ON e.course_id = c.course_id
        GROUP BY c.course_id
        HAVING COUNT(*) = (
                            SELECT MAX(EnrollmentCount)
                            FROM (
                            SELECT course_id, COUNT(*) AS EnrollmentCount
                            FROM enrollments
                            GROUP BY course_id
                            ) AS SubQuery
                        )
        ORDER BY NumberOfEnrollments DESC;      
        """,
            },
            {
                "role": "user",
                "content": f"### Complete SQLite SQL query only and with no explanation.\n\n### SQLite SQL tables, with their properties:\n#\n# {schema_metadata}\n#\n### {question}\nSELECT",
            },
        ],
    )
    sql = completion.choices[0].message.content.strip()
    return sql


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