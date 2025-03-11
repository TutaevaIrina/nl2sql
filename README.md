# NL2SQL

## **Overview**
This project evaluates **Text-to-SQL generation** using various **prompting techniques** and **Retrieval-Augmented Generation (RAG) architecture** with **Large Language Model (LLM)** **GPT-4o**.  
It is based on the **Spider dataset** (https://yale-lily.github.io/spider), a widely used benchmark for SQL generation from natural language.

The **Spider benchmark** is described in the following research paper:  
**"Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"**  
**[Paper link](https://arxiv.org/abs/1809.08887)**

---

## **Project Structure**
The project is organized into two main directories:  

- The **`prompt_strategies/`**  contains different **prompting techniques** for SQL query generation.    
- The **`rag/`** directory implements a **RAG pipeline**, which retrieves relevant database schema information before generating SQL queries.

---

## **Description of `prompt_strategies/`**
The **`prompt_strategies/`** directory contains four different **prompting techniques** used for SQL query generation:

### **1. basic_prompt/**
- Implements **Zero-Shot Prompting**, where the LLM is given only the **question and database schema**.
- **`generate_sql.py`**: Generates SQL queries from natural language.
- **`gold_example.txt`**: Contains ground-truth SQL queries from the Spider dataset.
- **`pred_example.txt`**: Stores the SQL queries generated by the LLM.

### **2. few_shot_prompt/**
- Uses **Few-Shot Prompting**, where the model is shown some **example SQL queries** to improve performance.
- **`generate_sql_few_shot_prompt.py`**: Generates SQL using few-shot examples.
- **`gold_example.txt`**: Contains ground-truth SQL queries from the Spider dataset.
- **`pred_example_basic_user_prompt.txt`**: Contains the SQL queries generated using this approach.

### **3. enhanced_few_shot_prompt/**
- Implements **Enhanced Few-Shot Prompting**, where example SQL queries are provided as context to the model and also instructions.
- **`generate_sql_enhanced_few_shot_prompt.py`**: Generates SQL using an improved few-shot prompt.
- **`gold_example.txt`**: Contains ground-truth SQL queries from the Spider dataset.
- **`match_log_enhanced_few_shot_prompt.txt`** & **`mismatch_log_enhanced_few_shot_prompt.txt`**: Log correct and incorrect queries.

### **4. zero_shot_prompt/**
- Uses Zero-Shot Prompting, where the model is given precise instructions without any examples to test whether it can generate SQL queries based solely on the instructions.
- **`generate_sql_zero_shot_prompt.py`**: Generates SQL using zero-shot instructions.
- **`gold_example.txt`**: Contains ground-truth SQL queries from the Spider dataset.
- **`pred_example_zero_shot_prompt.txt`**: Contains the SQL queries generated using this approach.

---

## **Description of `rag/`**
The **`rag/`** directory implements **Retrieval-Augmented Generation (RAG)** for SQL query generation.  
It retrieves relevant database schema information using **FAISS** and generates SQL queries with an LLM **GPT-4o**.

### **Key Components:**
- **`data_processor.py`** – Extracts database schema information and stores it in FAISS.  
- **`retriever.py`** – Retrieves relevant schema data from FAISS and sends it to the LLM for query generation.  
- **`faiss_index.idx`** – FAISS index storing schema embeddings.  
- **`faiss_text_data.json`** – JSON file storing the raw schema text data.  
- **`questions.txt`** – Contains user questions for SQL generation.  
- **`pred_example.txt`** – Stores the generated SQL queries for evaluation. 

---

## **How Does This Project Work?**

This project supports two main approaches for **Text-to-SQL generation**:  
- **Prompt-Based Approaches** (Zero-Shot, Few-Shot, and Enhanced Few-Shot Prompting)  
- **Retrieval-Augmented Generation (RAG)**  

---

### **1. Load questions and database schema**
- **Questions** are stored in **`questions.txt`** (or **`questions_experiment.txt`**).  
- **Schema information** is extracted from **`tables.json`** from the Spider Dataset (https://yale-lily.github.io/spider).  
- In **prompt-based approaches**, the schema is **directly included in the prompt**.  
- In **RAG**, the schema is **embedded and stored in FAISS for retrieval**.  

---

### **2. Generate SQL using LLMs**

#### **Prompt-Based Approaches (`prompt_strategies/`)**
- Uses **Zero-Shot, Few-Shot, or Enhanced Few-Shot Prompting**.  
- The model receives:  
  - A **SQL question**  
  - The **database schema**  
  - (Optionally) **example SQL queries** for better generalization  

#### **RAG-Based SQL Generation (`rag/`)**
- Uses **FAISS** to retrieve the most relevant schema information.  
- The LLM receives:  
  - A **SQL question**  
  - The **retrieved schema from FAISS**  
- This allows **handling larger schemas** without overloading the prompt.  

---

### **3. Evaluate Generated SQL Queries**
Regardless of whether SQL queries are generated using **prompting techniques** or **RAG-based retrieval**, the evaluation process follows the same approach:

- **Compare the generated SQL with ground-truth queries (`gold_example.txt`)**.  
- **Use evaluation scripts (`evaluation.py`)** to measure accuracy.  
- **Log errors and mismatches in `mismatch_log.txt`**.
- **Log successful matches in `match_log.txt`**.

---

## **Key Differences Between Prompting and RAG**
| Feature                | `prompt_strategies/` (Prompting)           | `rag/` (RAG)                         |
|------------------------|--------------------------------------------|--------------------------------------|
| **Schema Handling**    | Entire schema included in prompt           | Schema retrieved using FAISS         |
| **Query Context**      | Can include examples for few-shot learning | Uses retrieval to limit context size |
| **Best For**           | Small to medium-sized schemas              | Large databases with many tables     |
| **Limitations**        | Context length constraints and costs       | FAISS may retrieve incorrect or      |
|                        |                                            | incomplete schema                    |

---

## **Evaluation Results**
This section presents the accuracy of generated SQL queries using **both prompting-based approaches and RAG-based retrieval**.

### **1. Evaluation Metrics**
- **Execution Accuracy (EX)**: Percentage of generated SQL queries that **return the correct results** when executed on the database.

### **2. Results Overview**
| Approach                           | Execution Accuracy (%)   | 
|------------------------------------|--------------------------|
| **Basic Prompting**                | 76,4%                    | 
| **RAG-Based SQL Generation**       | 73,4%                    |


### **3. Scope of the Evaluation**
**Why are results only shown for Basic Prompt and RAG?**  
- The **same queries** were used for both approaches, making them **directly comparable**.  
- Other prompting strategies (**Few-Shot, Enhanced Few-Shot, Zero-Shot**) use different input queries and are therefore not included in this comparison.  

**How was the evaluation conducted?**  
- The reported results come from the **automated evaluation** based on the provided **evaluation script (`evaluation.py`)**.  
- A **separate manual evaluation** was also conducted, but these results are **not included here**.  
- The **manual evaluation** assessed query correctness beyond execution accuracy, considering logical equivalence and meaningful variations in query structure.  

---

## **NLTK Tokenizer Models: Manual Installation**
The project uses the **NLTK `punkt` tokenizer** by evaluating for text processing.  
In some cases, automatic installation via `nltk.download('punkt')` may fail due to network restrictions.  

To manually install the required tokenizer models, follow these steps:  

### **1. Download Required Packages**
Download the following tokenizer models from the official **NLTK repository**:  
🔗 **[NLTK Data - Tokenizers](https://www.nltk.org/nltk_data/)**
  
- **Punkt Tokenizer Models (`punkt.zip`)**  
  - [Download `punkt.zip`](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip)  
- **Punkt Tokenizer Models (`punkt_tab.zip`)**  
  - [Download `punkt_tab.zip`](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip)  


### **2. Extract the Files to the Correct Location**
Once downloaded, extract the **`punkt.zip`** and **`punkt_tab.zip`** files
   