import requests
import re
from ddgs import DDGS
from newspaper import Article

OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_MODEL = "llama3.2:3b"
CRITIC_MODEL = "qwen2.5:1.5b"


# -----------------------------
# OLLAMA CALL
# -----------------------------
def ollama_call(model, prompt):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload)

    return r.json()["response"]


# -----------------------------
# EVIDENCE RETRIEVAL
# -----------------------------
TRUSTED_DOMAINS = [
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "thehindu.com",
    "indianexpress.com",
    "altnews.in",
    "ndtv.com",
    "theguardian.com",
    "bloomberg.com",
    "washingtonpost.com",
    "afp.com"
]



def retrieve_evidence(query, max_results=10):

    evidence_text = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=15)

    for r in results:

        url = r.get("href", "")

        try:

            article = Article(url)
            article.download()
            article.parse()

            snippet = article.text[:1000]

            if snippet.strip():
                evidence_text.append(snippet)

        except:
            continue

        if len(evidence_text) >= max_results:
            break

    if len(evidence_text) == 0:
        return "No strong external evidence retrieved."

    combined = "\n\n".join(evidence_text[:10])

    return combined



# -----------------------------
# CONFIDENCE EXTRACTION
# -----------------------------

def extract_confidence(text):

    match = re.search(
        r"Initial Confidence:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


# -----------------------------
# JUDGE PROMPTS
# -----------------------------
def judge_initial(article):

    prompt = f"""
You are a neutral fact-checking analyst.

Read the news article and provide reasoning about whether it appears real or fake.

Focus on:
- logical consistency
- sensational language
- unsupported claims
- plausibility

At the end output:

Initial Confidence: <number between 0.00 and 1.00>

ARTICLE:
{article}

Output format:

Reasoning: <your reasoning>

Initial Confidence: <number>
"""

    return ollama_call(JUDGE_MODEL, prompt)


def judge_revision(article, prev_reasoning, critic_feedback):

    prompt = f"""
You previously reasoned about a news article.

Previous reasoning:
{prev_reasoning}

Critic feedback:
{critic_feedback}

Revise your reasoning by correcting assumptions or weak logic.


ARTICLE:
{article}

Output revised reasoning only.
"""

    return ollama_call(JUDGE_MODEL, prompt)


def judge_final(article, prev_reasoning, critic_feedback, evidence):

    prompt = f"""
You are performing the final reasoning step.

Previous reasoning:
{prev_reasoning}

Critic feedback:
{critic_feedback}

External Evidence:
{evidence}

Using all information, produce final reasoning on whether the article is real or fake.

Do NOT output confidence.
Only reasoning.
"""

    return ollama_call(JUDGE_MODEL, prompt)


# -----------------------------
# CRITIC PROMPTS
# -----------------------------
def critic_review(reasoning):

    prompt = f"""
You are a critic reviewing reasoning.

Check for:
- hallucinations
- unsupported assumptions
- logical gaps
- overconfidence

Reasoning:
{reasoning}

Output ONLY one of the following:

ACTION: OK

or

ACTION: REEVALUATE

Then briefly explain why.

Focus ONLY on evaluating the quality of reasoning.
Do NOT independently classify the article.
"""

    return ollama_call(CRITIC_MODEL, prompt)



def critic_with_evidence(reasoning, evidence):

    prompt = f"""
    You are evaluating Judge reasoning against external evidence.

    Judge reasoning:
    {reasoning}

    External Evidence:
    {evidence}

    If evidence supports the reasoning, say "OK"

    If evidence contradicts the reasoning, say "REEVALUATE"

    Then briefly explain why.
    """

    return ollama_call(CRITIC_MODEL, prompt)

def critic_confidence(final_reasoning):

    prompt = f"""
Based ONLY on the reasoning below, output ONLY ONE number between 0.00 and 1.00.

No explanation.
No words.
Only the number.

Reasoning:
{final_reasoning}
"""

    output = ollama_call(CRITIC_MODEL, prompt)

    match = re.search(
    r"(0(?:\.\d+)?|1(?:\.0+)?)",
    output)

    if match:
        return float(match.group())

    return 0.5


# -----------------------------
# LABEL MAPPING
# -----------------------------
def label_from_confidence(conf):

    if conf > 0.50:
        return "REAL"
    else:
        return "FAKE"


# -----------------------------
# JC LOOP
# -----------------------------
def run_jc(article):

    # =========================================
    # ITERATION 1
    # =========================================
    print("\n=== ITERATION 1 ===")

    judge1 = judge_initial(article)

    print("\nJudge Output:\n")
    print(judge1)

    initial_conf = extract_confidence(judge1)

    # -----------------------------------------
    # EARLY EXIT MECHANISM
    # -----------------------------------------
    if initial_conf is not None:

        print("\nInitial Confidence:", initial_conf)

        if initial_conf >= 0.90:

            print("\nEARLY EXIT TRIGGERED")
            print("High confidence REAL prediction.")

            print("\nFinal Label: REAL")
            return

        elif initial_conf <= 0.10:

            print("\nEARLY EXIT TRIGGERED")
            print("High confidence FAKE prediction.")

            print("\nFinal Label: FAKE")
            return

    # -----------------------------------------
    # CRITIC REVIEW
    # -----------------------------------------
    critic1 = critic_review(judge1)

    print("\nCritic Output:\n")
    print(critic1)

    # =========================================
    # ITERATION 2
    # =========================================
    print("\n=== ITERATION 2 ===")

    judge2 = judge_revision(article, judge1, critic1)

    print("\nRevised Judge Reasoning:\n")
    print(judge2)

    # -----------------------------------------
    # RETRIEVE EVIDENCE
    # -----------------------------------------
    print("\nRetrieving external evidence...\n")

    evidence = retrieve_evidence(article)

    print("Evidence Retrieved.\n")

    critic2 = critic_with_evidence(judge2, evidence)

    print("\nCritic Output:\n")
    print(critic2)

    # -----------------------------------------
    # EARLY TERMINATION AFTER ITERATION 2
    # -----------------------------------------
    if "ACTION: OK" in critic2:

        print("\nReasoning accepted after Iteration 2.")

        final_conf = critic_confidence(judge2)

        final_label = label_from_confidence(final_conf)

        print("\nFinal Confidence:", final_conf)
        print("Final Label:", final_label)

        return

    # =========================================
    # ITERATION 3
    # =========================================
    print("\n=== ITERATION 3 ===")

    judge3 = judge_final(
        article,
        judge2,
        critic2,
        evidence
    )

    print("\nFinal Judge Reasoning:\n")
    print(judge3)

    # -----------------------------------------
    # FINAL CONFIDENCE
    # -----------------------------------------
    final_conf = critic_confidence(judge3)

    final_label = label_from_confidence(final_conf)

    print("\nFinal Confidence:", final_conf)
    print("Final Label:", final_label)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Paste news article.")
    print("Press CTRL+Z then Enter (Windows)")
    print("or CTRL+D (Linux/Mac)\n")

    lines = []

    while True:
        try:
            lines.append(input())
        except EOFError:
            break

    article = "\n".join(lines)

    run_jc(article)