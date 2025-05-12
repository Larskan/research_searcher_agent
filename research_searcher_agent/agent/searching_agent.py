from autogen import ConversableAgent
from autogen.oai.client import MistralAIClient
from research_searcher_agent.tools.query import query_handling, print_papers
from research_searcher_agent.config import LLM_CONFIG
import json
import os

Search_prompt = """
Use the parameters below to search using the API Endpoint described within 'query_handling'.

Parameters: {input}

Once you've found the results, display the results and reply with TERMINATE.
"""

def create_searching_agent() -> ConversableAgent:

    # Dictionary comprehension. Creates new dictionary that copies entire config..except api_rate_limit
    llm_config_clean={
            "config_list":[
                {k: v for k, v in LLM_CONFIG["config_list"][0].items() if k != "api_rate_limit"}
            ]
        }
    # define the agent
    agent = ConversableAgent(
        name="Research Searcher",
        system_message="""
        You are a research paper search assistant.

        1. Prompt the human to provide the following parameters: 
            - Title
            - Year filter (specify "in", "before", or "after" a given year)
            - Minimum citations (example: 100)
        
        2. Once all parameters are received, call the tool `query_handling(title, year, citations)` to search for papers.

        3. Find 5 papers.

        4. Use `print_papers()` to display the results to the human.

        5. Save the results in a list as a json file.

        6. End your reply with the word TERMINATED once the results have been shown.

        You may ask follow-up questions to clarify incomplete or ambigous input.
        """,
        llm_config=llm_config_clean,
    )

    # If we turn this off, then it cant give more than 2 answers, despite us asking for minimum 5.
    # Patch rate limit manually after initialization
    if isinstance(agent.client, MistralAIClient):
        agent.client.rate_limit = LLM_CONFIG["config_list"][0].get("api_rate_limit", None)

    # add the tools to the agent
    agent.register_for_llm(name="query_handling", description="Contains the queries for searching with the API")(query_handling)
    agent.register_for_llm(name="print_papers", description="Contains the queries for searching with the API")(print_papers)

    return agent



def create_user_proxy():
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="TERMINATE",
    )
    user_proxy.register_for_execution(name="query_handling")(query_handling)
    user_proxy.register_for_execution(name="print_papers")(print_papers)
    return user_proxy

def serialize_message(message):
        return{
                "role": message.get("role"),
                "name": message.get("name"),
                "content": message.get("content"),
                "function_call": message.get("function_call", None),
                "tool_calls": message.get("tool_calls", None),
        }

def main():
    user_proxy = create_user_proxy()
    searching_agent = create_searching_agent()
    # Getting user input
    topic = input("Enter the research title: ")
    year = input("Enter Before/After/In and the year: ")
    citations = input("Enter the citations: ")


    # Send user input to the agent as dynamic prompt
    user_input = f"Topic: {topic}, Year: {year}, Citations: {citations}"

    user_proxy.initiate_chat(
        searching_agent, 
        message=Search_prompt.format(input=user_input)
    )

    history = searching_agent.chat_messages.get(user_proxy, [])
    serialized_history = [serialize_message(m) for m in history]

    save_path = os.path.join(os.getcwd(), "full_convo_history.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serialized_history, f, indent=2, ensure_ascii=False)
    print("Full convo saved to full_convo_history.json")

if __name__ == "__main__":
    main()