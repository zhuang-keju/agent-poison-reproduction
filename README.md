# agent-poison-reproduction
# Red-Teaming RAG Agents: Simulation of Knowledge Base Poisoning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/release/python-3100/)
[![RAG Framework](https://img.shields.io/badge/RAG-LangChain%20%7C%20LlamaIndex-orange)]()

> **Note:** This repository is an active research project reproducing the **"AgentPoison"** attack vector described in *Chen et al. (2024)*.
>
> It serves as a testbed for developing robust RAG (Retrieval-Augmented Generation) systems, specifically focusing on validating the proposed **Attribution-Based Defense** mechanism to mitigate knowledge base poisoning.

## 📄 Abstract

Retrieval-Augmented Generation (RAG) is the backbone of modern LLM Agents. However, RAG systems are vulnerable to **Knowledge Base Poisoning**, where attackers inject optimized adversarial texts into the retrieval corpus.

This project reproduces the **AgentPoison** attack, which optimizes malicious text segments to have high semantic similarity with specific target queries (triggers) in the embedding space. When a user asks a triggered question, the RAG system retrieves the poisoned context, causing the agent to execute malicious instructions.

**Reference Paper:**
*Chen, Z., et al. (2024). AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. arXiv preprint.* [[Paper Link]](https://arxiv.org/abs/2407.12784)

## 🏗️ Architecture & Methodology

The project consists of two main modules:
1.  **The Attacker (Poison Generator):** Uses gradient-guided search / greedy optimization to create adversarial text passages that maximize the dot-product similarity with a target query vector while minimizing semantic suspicion (Uniqueness/Compactness loss).
2.  **The Defender (Attribution Validator):** A proposed module that cross-references the generated response against the retrieved chunks to detect hallucinated or injected malicious commands.

---

## 🗺️ Implementation Roadmap

Current development focus: **Stage 2 (Poison Generation)**.

- [x] **Phase 1: RAG Pipeline Setup**
    - [x] Environment setup: PyTorch, LangChain, FAISS (Vector DB).
    - [x] Data ingestion: Preprocessing the benign corpus (WikiText/MS MARCO).
    - [x] Baseline RAG: Implementing a standard dense retrieval pipeline using `sentence-transformers` (e.g., Contriever/MPNet).

- [ ] **Phase 2: Poison Injection (The Attack)**
    - [ ] **Trigger definition:** Defining target queries (e.g., "Summarize the financial report") and malicious payloads (e.g., "transfer funds").
    - [ ] **Optimization algorithm:** Implementing the search heuristic to generate poison text that maps to the trigger's embedding neighborhood.
    - [ ] Injection: Inserting poison chunks into the FAISS index.

- [ ] **Phase 3: Defense Implementation (Research Goal)**
    - [ ] **Attribution Check:** Implementing the "Answer-to-Source" validation logic as proposed in my research framework.
    - [ ] Evaluation: Benchmarking Attack Success Rate (ASR) vs. Defense Drop Rate.

---

## 📂 Repository Structure

```text
├── data/
│   ├── corpus_benign/       # Clean knowledge base documents
│   └── triggers/            # Target triggers and intended malicious actions
├── src/
│   ├── rag_pipeline.py      # Standard RAG implementation (LangChain)
│   ├── attack_optimizer.py  # Algorithms for generating poisoned embeddings
│   └── defense_verifier.py  # Attribution-based consistency checks
├── notebooks/
│   ├── 01_baseline_rag.ipynb    # Demo of the clean system
│   └── 02_poison_simulation.ipynb # (WIP) Demonstrating the attack
├── requirements.txt
└── README.md

```

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **RAG Framework:** LangChain / LlamaIndex
* **Vector Database:** FAISS (Facebook AI Similarity Search) or ChromaDB
* **Embeddings:** Hugging Face `sentence-transformers` (all-mpnet-base-v2)
* **LLM:** Llama-3-8B (via Ollama or Hugging Face) / OpenAI API

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/zhuang-keju/agent-poison-reproduction
cd agent-poison-reproduction
pip install -r requirements.txt

```

### Usage (Baseline)

*To run the benign RAG pipeline:*

```bash
python src/rag_pipeline.py --query "What is the capital of France?"

```

*(Poison generation scripts will be released in the next update).*

## 🤝 Contribution & Contact

This project is maintained by **Zhuang Keju** as part of an independent research initiative on Trustworthy AI Agents.

* **Email:** zhuangkeju@gmail.com
* **GitHub:** https://github.com/zhuang-keju

---

*Disclaimer: This repository is for academic research and defensive testing purposes only.*

