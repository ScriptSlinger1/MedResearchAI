# 🧬 MedResearch AI

> **AI-powered medical literature research, built with Django.**

MedResearch AI is a web application designed to help users explore medical research more efficiently. Instead of asking an AI model to generate medical information from memory, the system is designed around **retrieval, evidence, and citations**.

The goal is simple:

**Ask a medical research question → find relevant scientific literature → analyze the evidence → return a structured, source-backed answer.**

> [!WARNING]
> **Research tool only.** MedResearch AI is not a medical device, diagnostic system, or substitute for a qualified healthcare professional. It should not be used to make clinical decisions or provide individualized medical advice.

---

## ✨ Features

### 🔎 Evidence Retrieval
Search relevant medical literature and identify sources that can support an answer.

### 🤖 AI-Assisted Analysis
Use an LLM to synthesize retrieved research into a clear, structured response.

### 📚 Source-Backed Answers
Connect generated claims to the research used to produce them instead of presenting unsupported AI output.

### 📊 Evidence-Aware Results
Surface important information such as:

- Key findings
- Study limitations
- Evidence strength
- Conflicting results
- Relevant sources

### 💾 Research History
Save previous questions and results so users can return to their research later.

### 👤 User Accounts
Authenticated users can maintain their own research history and saved literature.

---

## 🧠 How It Works

MedResearch AI follows a **Retrieval-Augmented Generation (RAG)** architecture.

```text
┌───────────────────┐
│       User        │
│  Medical Question │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│      Django       │
│    API / Backend  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Literature Search │
│    & Retrieval    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Relevant Research │
│      Papers       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    AI Analysis    │
│  & Summarization  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Structured Answer  │
│ + Evidence +      │
│     Citations     │
└───────────────────┘