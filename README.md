# NL2SQL

## **Overview**
This project evaluates **Text-to-SQL generation** using various **prompting techniques** for **Large Language Model (LLM) - GPT-4o**. 
It is based on the **Spider dataset** (https://yale-lily.github.io/spider), a widely used benchmark for SQL generation from natural language.

## **Project Structure**
The project consists of multiple directories, each implementing a different **prompting strategy**:


## **Directory Descriptions**
### **1. basic_prompt/**
- Implements **Zero-Shot Prompting**, where the LLM is given only the **question and database schema**.
- **`generate_sql.py`**: Generates SQL queries from natural language.
- **`gold_example.txt`**: Contains ground-truth SQL queries from the Spider dataset.
- **`pred_example.txt`**: Stores the SQL queries generated by the LLM.
- **`mismatch_log.txt`**: Logs mismatches between **gold queries** and **predicted queries**.

### **2. few_shot_prompt/**
- Uses **Few-Shot Prompting**, where the model is shown some **example SQL queries** to improve performance.
- **`generate_sql_few_shot_prompt.py`**: Generates SQL using few-shot examples.
- **`pred_example_basic_user_prompt.txt`**: Contains the SQL queries generated using this approach.

### **3. enhanced_few_shot_prompt/**
- Implements **Enhanced Few-Shot Prompting**, where example SQL queries are provided as context to the model and also instructions.
- **`generate_sql_enhanced_few_shot_prompt.py`**: Generates SQL using an improved few-shot prompt.
- **`match_log_enhanced_few_shot_prompt.txt`** & **`mismatch_log_enhanced_few_shot_prompt.txt`**: Log correct and incorrect queries.

### **4. zero_shot_prompt/**
- Uses Zero-Shot Prompting, where the model is given precise instructions without any examples to test whether it can generate SQL queries based solely on the instructions.
- generate_sql_zero_shot_prompt.py: Generates SQL using zero-shot instructions.
- pred_example_zero_shot_prompt.txt: Contains the SQL queries generated using this approach.

---

## **How Does This Project Work?**
1. **Load questions and database schema.**
   - Questions are stored in **`questions.txt`** (or **`questions_experiment.txt`**).
   - Schema information is extracted from **`tables.json`** from the Spider Dataset (https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view).

2. **Generate SQL using LLMs.**
   - Uses **Zero-Shot**, **Few-Shot**, or **Enhanced Few-Shot Prompting**.
   - The model receives a **SQL question**, the **database schema**, and optionally **example SQL queries**.

3. **Evaluate generated SQL queries.**
   - **Compare with ground-truth queries (`gold_example.txt`)**.
   - **Evaluation scripts (`evaluation.py`)** measure accuracy.
   - **Errors are logged in `mismatch_log.txt`**. 
   