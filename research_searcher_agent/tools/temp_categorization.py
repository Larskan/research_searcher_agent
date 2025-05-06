from autogen import AssistantAgent
from research_searcher_agent.config import LLM_CONFIG
import json

# Class Purpose: Wraps around a LLM call to parse/categorize a freeform string into structures parameters. 

def categorize_research(text: str) -> str:
    agent = AssistantAgent(
        name="Research Agent",
        system_message="""
        You extract research search parameters from user input. 
        Return a JSON object with the following keys:
        - title (string)
        - year (string: e.g. "in 2020", "before 2010", "after 2015")
        - citations (integer)

        Respond ONLY with a JSON object, no extra text.
        """,
        llm_config=LLM_CONFIG,
    )
    reply = agent.generate_reply(messages=[{"role": "user", "content": text}])

    try:
        return json.loads(reply["content"])
    except Exception:
        raise ValueError(f"Could not parse input: {reply}")
