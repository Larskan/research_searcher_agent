from autogen import AssistantAgent
from research_searcher_agent.config import LLM_CONFIG


Search_prompt = """
Search for the result as best you can. You have access to the tools provided.
Request the parameters of topic, year and citations from the human: {input}
Use the parameters to search using the API Endpoint described within 'query_handling'

Once you've found the results, display the results and reply with TERMINATE.
"""

def categorize_research(text: str) -> str:
    agent = AssistantAgent(
        name="Research Agent",
        system_message=Search_prompt,
        llm_config=LLM_CONFIG,
    )
    reply = agent.generate_reply(
        messages=[
            {"role": "user", "content": f'answer to the request: {text}'}
        ],
    )

    if not reply:
        raise ValueError("No reply found")

    reply_value = ""
    if isinstance(reply, dict):
        reply_content = reply["content"]
        if reply_content:
            reply_value = reply_content
        else:
            raise ValueError("No content found in the reply")
    else:
        reply_value = reply

    reply_values = reply_value.splitlines()
    if len(reply_values) != 1:
        reply_value = reply_values[0]

    reply_value = reply_value.replace("[", "").replace("]", "").replace(" ", "").strip()

    return reply_value