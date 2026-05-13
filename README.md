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
---
## Clone Repository
git clone https://github.com/meerbose2001/JC_MTP_THESIS.git
cd JC_MTP_THESIS
## Install Dependencies
pip install -r requirements.txt
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
python JC_retrieval.py
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
C:\Users\SWARNAVA\Documents\JC FAKE NEWS  RETRIVAL>python JC_Retrieval.py
Paste news article.
Press CTRL+Z then Enter (Windows)
or CTRL+D (Linux/Mac)

The Greater Chennai Corporation (GCC) has reported spending Rs 1,692 crore on roadworks since 2021, but many residents, particularly in North Chennai and newly added areas, still encounter poorly maintained roads.

^Z

=== ITERATION 1 ===

Judge Output:

Reasoning:
The article appears to be based on real information about the Greater Chennai Corporation's spending on roadworks. The specific figure of Rs 1,692 crore is a credible amount and can be easily verified through official records or news reports from reputable sources.

However, the assertion that many residents in North Chennai and newly added areas still encounter poorly maintained roads might be exaggerated or an oversimplification. While it's possible that some residents are facing issues with road maintenance, it's unlikely that the entire area is plagued by poor conditions.

The language used is objective and factual, which suggests that the article is not sensationalized or biased. There are no glaring red flags or obvious attempts to mislead readers.

One potential issue is the lack of specific data on how much of the Rs 1,692 crore has been spent in North Chennai and newly added areas, which could indicate a mismatch between reported spending and actual implementation.

Initial Confidence: 0.85

Initial Confidence: 0.85

Critic Output:

ACTION: REEVALUATE

Explanation: The reasoning does not sufficiently address several critical aspects that should be considered for a more accurate and nuanced understanding. While it acknowledges that some areas may face poor road maintenance, it fails to provide clear or concrete evidence of this issue across all affected neighborhoods within North Chennai and newly added areas. Additionally, the lack of specific data on spending allocation can lead to skepticism about the article's claims regarding actual improvements in road conditions. Therefore, further verification and clarification are needed to make a more informed assessment.

=== ITERATION 2 ===

Revised Judge Reasoning:

Reasoning:

While the article presents a reasonable summary of the GCC's expenditure on roadworks, several aspects warrant reevaluation. The reported spending figure of Rs 1,692 crore appears credible, but the claim that many residents in North Chennai and newly added areas face poorly maintained roads requires closer examination.

The language used is still objective and factual, indicating a neutral tone. However, this neutrality does not necessarily imply a lack of controversy or potential bias in the reporting. The absence of specific data on spending allocation for North Chennai and newly added areas raises concerns about the accuracy of the article's claims regarding actual improvements in road conditions.

To improve confidence in the article's assertions, it would be beneficial to verify the existence of concrete evidence, such as resident feedback surveys, photographs or videos of poorly maintained roads, or interviews with city officials about their efforts to address these issues. Further scrutiny is needed to determine whether the reported problem persists across all affected neighborhoods within North Chennai and newly added areas.

Initial Confidence: 0.70

Retrieving external evidence...

Evidence Retrieved.


Critic Output:

**REVIEW EVALUATION**

The external evidence provided paints a more detailed and specific picture of the issues faced by residents in Chennai regarding road maintenance compared to the initial article's summary. The evidence highlights that:

1. **Incomplete Spending Reports**: Despite spending 1,692 crore on roadworks, many roads have not been relaid as claimed, indicating possible discrepancies between reported expenditures and actual improvements.

2. **Poor Condition of Existing Roads**: A significant portion of the roads in North Chennai and newly added areas remain poorly maintained, with residents facing issues like potholes and damaged surfaces.

3. **Efforts to Address Issues**: The evidence shows that while some efforts are being made through projects like relaying 15,107 roads worth 1,692 crore under various schemes, these have not been completed on a timely basis, particularly in the area of relaying interior roads.

4. **Specific Projects and Claims**: It clarifies that specific details regarding road works (like those involving NSMT, TURIF, and GCC funds) are provided but do not indicate concrete evidence for improvements.

5. **Future Work Plans**: The article mentions projects to monitor quality controllers and improve footpaths, suggesting ongoing efforts to address these issues in the future.

**Conclusion:**
The external evidence provides a more nuanced understanding of the Chennai road situation compared to the initial summary. It shows that while there have been some improvements (like 15,107 roads completed), significant areas remain unaddressed due to financial and implementation challenges. This indicates that efforts have not kept pace with the stated expenditures, warranting a reevaluation of reported progress and funding allocation for road projects in Chennai.

=== ITERATION 3 ===

Final Judge Reasoning:

Based on the provided information, the article appears to be largely factual. The language used is formal and objective, with specific details about roadwork expenditures, project plans, and public complaints. While some statements are anecdotal (e.g., from a resident of Kottivakkam), they are contextualized within the broader scope of city-wide issues.

Several indicators suggest that the article may be based on real events and concerns:

1. Specific numbers: The article provides concrete figures for roadwork expenditures, project allocations, and completed works, which suggests an effort to provide accurate information.
2. Local government sources: Quotes from GCC officials and other stakeholders add credibility to the report.
3. External references: The inclusion of external evidence (e.g., articles or reports) about Chennai's infrastructure challenges supports the article's claims.

However, some aspects raise potential red flags:

1. Lack of primary sources: While the article includes quotes from GCC officials and residents, there is limited direct evidence from city officials or government agencies.
2. Unverifiable claims: Some statements (e.g., about specific projects or funding allocations) could be difficult to verify without additional information.
3. Overemphasis on negative aspects: The article focuses primarily on infrastructure issues, which may create a skewed representation of the city's overall situation.

To further assess the article's authenticity, it would be necessary to cross-reference claims with other credible sources and explore primary documentation from local government agencies or official records.

Final Confidence: 0.95
Final Label: REAL

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
