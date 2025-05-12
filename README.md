# Research searching agent and evaluation

This is an AI Agent that is focused around researching topics through a research paper database and evaluating the result.

## Setup

Install the Python dependencies.

```bash
pip install -r requirements.txt
```
Activate your virtual environment.

## Run the Searching Agent

```bash
python -m research_searcher_agent.agent.searching_agent --mode query_handling
```

## Run the Evaluation Agent
```bash
python -m research_searcher_agent.agent.evaluation_agent
```

## Requirements

- Python 3.10+
- autogen
- ollama
- fix-busted-json

> [!NOTE]
> Open source LLM's need to support tool calling for this to work.
> LLama 3.1 and LLama 3.1 Instruct support tool calling.
