# A Judge Critic Multi-Agent LLM System for Zero-Shot Fake News Detection

## Overview

This project presents a hybrid multi-model Judge–Critic (JC) framework for zero-shot fake news detection using Large Language Models (LLMs). The system combines structured reasoning, adversarial critique, adaptive iteration control, and external evidence verification to analyze the credibility of news articles without requiring supervised training or fine-tuning.

The framework is designed around two heterogeneous local LLMs:

* **Llama 3.2** acts as the **Judge**
* **Qwen 2.5** acts as the **Critic**

The Judge generates structured reasoning regarding the plausibility and credibility of an article, while the Critic evaluates the Judge’s reasoning for logical inconsistencies, unsupported assumptions, hallucinations, and weak justification. Through iterative critique and refinement, the framework progressively improves reasoning quality before generating a final confidence score and prediction label.

Unlike conventional fake news detection systems that rely heavily on labeled datasets and supervised classifiers, this framework focuses on reasoning-driven verification and evidence-grounded critique in a zero-shot setting.

---

# Key Features

* Zero-shot fake news detection using local LLMs
* Hybrid multi-agent Judge–Critic architecture
* Adaptive 1–3 iteration reasoning pipeline
* Early confidence exit for computational efficiency
* External evidence retrieval from trusted news domains
* Evidence-grounded reasoning and verification
* Numeric confidence score generation (0.00–1.00)
* Iterative self-correcting reasoning mechanism
* Reduced hallucination through adversarial critique
* Fully local deployment using Ollama
* Interpretable reasoning traces and explanations

---

# Problem Statement

Traditional fake news detection systems typically depend on:

* Large labeled datasets
* Supervised model training
* Retrieval-heavy fact-checking pipelines

However, these approaches often struggle with:

* Generalization to unseen topics
* High data requirements
* Expensive infrastructure
* Lack of interpretability
* Hallucinations and overconfidence in LLM outputs

This project addresses these limitations by introducing a collaborative multi-agent reasoning framework where one model reasons about article credibility while another model critiques and refines the reasoning iteratively.

---

# System Architecture

The proposed framework consists of three primary components:

## 1. Judge (Llama 3.2)

The Judge analyzes the news article and produces:

* Initial reasoning
* Early confidence score (only in Iteration 1)

The Judge focuses on:

* Narrative consistency
* Plausibility of claims
* Linguistic tone
* Contextual coherence
* Suspicious or exaggerated statements

---

## 2. Critic (Qwen 2.5)

The Critic evaluates the Judge’s reasoning for:

* Logical inconsistencies
* Unsupported assumptions
* Hallucinated facts
* Weak justification
* Contradictions with external evidence

The Critic outputs:

* ACTION: OK
* ACTION: REEVALUATE

In the final stage, the Critic also generates a calibrated numeric confidence score between 0.00 and 1.00.

---

## 3. Evidence Retrieval Module

For uncertain predictions, the framework retrieves external evidence from trusted news sources such as:

* BBC
* Reuters
* AP News
* The Hindu
* Indian Express
* Alt News
* NDTV
* Bloomberg
* AFP Fact Check
* The Guardian

Retrieved articles are summarized and injected into later reasoning stages for factual grounding.

---

# Adaptive Iterative Reasoning Pipeline

The system operates using an adaptive 1–3 iteration reasoning mechanism.

## Iteration 1

* Judge produces initial reasoning and early confidence score.
* If confidence ≥ 0.90 or ≤ 0.10:

  * System exits immediately.
  * Evidence retrieval and additional iterations are skipped.
  * Prediction behaves as a fast single-pass LLM evaluation.

## Iteration 2

* Triggered only for uncertain cases.
* Critic evaluates Judge reasoning.
* Judge revises reasoning based on Critic feedback.
* External evidence is retrieved and evaluated.

## Iteration 3

* Activated only if contradictions remain.
* Judge receives:

  * Original article
  * Previous reasoning
  * Critic feedback
  * Retrieved evidence
* Final refined reasoning is generated.
* Critic outputs final confidence score.

---

# Dataset

The experiments were conducted on a custom fake news dataset consisting of real and fake news articles collected from:

* Indian news outlets
* Fact-checking platforms
* Trusted media sources

The dataset includes:

* Article titles
* Article content
* Real/Fake labels

The title and content are merged into a single contextual text sequence before being passed into the reasoning pipeline.

---

# Experimental Results

The proposed JC framework demonstrated strong performance in a zero-shot setting.

## Performance Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.81  |
| Precision | 0.821  |
| Recall    | 0.7925  |
| F1 Score  | 0.807  |

The results indicate that structured multi-agent reasoning combined with evidence verification can substantially improve robustness compared to single-pass LLM predictions.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com/download

---

## Pull Required Models

```bash
ollama pull llama3.2
ollama pull qwen2.5
```

---

# Running the Project

## Run Main Pipeline

```bash
python src/main.py
```

---

## Example Execution

```bash
python src/main.py --input examples/sample_input.txt
```

---

# Dependencies

Main dependencies include:

* Python 3.10+
* pandas
* numpy
* scikit-learn
* requests
* beautifulsoup4
* openpyxl
* matplotlib
* ollama

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Output Format

The framework produces:

* Judge reasoning
* Critic feedback
* External evidence summary
* Final confidence score
* Final prediction label (Real/Fake)

Example:

```text
Judge Reasoning:
"The article references official reports and uses neutral language."

Critic Feedback:
"ACTION: OK"

Confidence Score:
0.82

Final Prediction:
REAL
```

---

# Advantages of the Framework

* Does not require supervised training
* Produces interpretable reasoning traces
* Reduces hallucinations using adversarial critique
* Adaptive iteration reduces computational cost
* Evidence grounding improves factual reliability
* Fully local deployment preserves privacy
* Modular architecture allows future extension

---

# Limitations

* Computationally expensive for multi-iteration cases
* Retrieval quality affects reasoning quality
* Limited multilingual support
* Sensitive to prompt engineering
* May struggle with sarcasm and implicit claims

---

# Future Work

Future work may expand the evidence retrieval system to support multi-source and multi-document reasoning with reliability-aware ranking of information sources. Another promising direction involves improving the Critic through adversarial prompting or fine-tuning techniques so that it can better identify subtle hallucinations, sarcasm, misleading narratives, and implicit claims. Further optimization of the adaptive iteration mechanism may also reduce computational cost by learning more effective stopping criteria and retrieval-triggering strategies. Finally, future studies may explore dynamic role assignment between language models and learning-based iteration control to further improve the efficiency and reliability of collaborative reasoning systems.

---

# Research Contribution

This work demonstrates that collaborative large language model systems can improve reliability in automated fact-checking through:

* Structured reasoning
* Adversarial critique
* Adaptive iterative refinement
* Evidence-grounded verification

The project highlights the potential of multi-agent LLM architectures for interpretable and robust misinformation detection in zero-shot settings.

---

# Citation

If you use this work in your research, please cite appropriately.

---

# Author

Swarnava Bose
M.Tech, Computer Science and Engineering
Indian Institute of Technology Bhubaneswar

---

# License

This project is intended for academic and research purposes.
