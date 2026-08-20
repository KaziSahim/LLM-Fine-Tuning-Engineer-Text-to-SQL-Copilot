# LLM-Fine-Tuning-Engineer-Text-to-SQL-Copilot
# 📊 Text-to-SQL Fine-Tuning with Qwen 2.5 7B

A high-performance implementation for fine-tuning **Qwen2.5-7B-Instruct** on natural language to SQL translation. Optimized for memory-constrained environments using **QLoRA** and **Unsloth**, enabling efficient training on standard T4 GPU instances.

---

## 🚀 Key Highlights

* **⚡ Ultra-Efficient Fine-Tuning:** Uses 4-bit quantization and Unsloth gradient checkpointing to drastically cut memory usage.


* **🎯 High Execution Accuracy:** Achieves an overall execution accuracy of **78.0%** and a **93.0%** valid query rate.
* **🛠️ Open-Source Stack:** Built using Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes`.



---

## 🛠️ Architecture & Tech Stack

| Component | Choice / Setting |
| --- | --- |
| **Base Model** 🧠 | `Qwen/Qwen2.5-7B-Instruct`<br> |
| **Quantization** 💎 | 4-bit NormalFloat (`load_in_4bit=True`)

 |
| **Fine-Tuning Method** 🎨 | QLoRA ($r=16$, $\alpha=16$)

 |
| **Target Modules** 🎯 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`<br> |
| **Dataset** 📚 | `b-mc2/sql-create-context`<br> |
| **Sequence Length** 📏 | 2048 tokens

 |
| **Optimizer** ⚡ | 8-bit AdamW (`adamw_8bit`)

 |

---

## 📈 Model Evaluation Dashboard

### 📊 Performance Breakdown

* **Overall Metrics:**
* **Valid Query Rate:** `93.0%`
* **Execution Accuracy:** `78.0%`
* **Exact Match:** `55.0%`


* **Execution Accuracy by Difficulty:**
* 🟢 **Easy:** `76.1%`
* 🟡 **Medium:** `72.2%`
* 🔴 **Hard:** `83.3%`


* **Query Output Errors:**
* ✅ **Correct:** `81.0%`
* ❌ **Logic Error:** `8.0%`
* ⚠️ **Syntax Error:** `6.0%`
* 🚫 **Schema Error:** `5.0%`



---

## ⚙️ Installation & Setup

1. **Install required dependencies:**

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes datasets
```[cite: 1]


```


2. **Authenticate with Hugging Face:**


Make sure to set your `HUGGINGFACE_TOKEN` in your environment or Colab secrets.



---

## 💻 Usage Example

```python
from unsloth import FastLanguageModel

# Load fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

# Switch to fast inference mode
FastLanguageModel.for_inference(model)

# Prompt layout
prompt_template = """You are an expert in SQL. Given the database schema, write the correct SQL query to answer the question.

### Schema:
{}

### Question:
{}

### SQL Query:
"""

schema = "CREATE TABLE employees (id INT, name VARCHAR, salary INT);"
question = "Find the names of employees earning more than 50000."

inputs = tokenizer(
    [prompt_template.format(schema, question, "")], 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```[cite: 1]

```

# Result
![](https://www.github.com/KaziSahim/LLM-Fine-Tuning-Engineer-Text-to-SQL-Copilot/result.png)
![](https://github.com/KaziSahim/LLM-Fine-Tuning-Engineer-Text-to-SQL-Copilot/blob/db8463cf927a049907f46b8e71e99b0998195cf6/result.png)
