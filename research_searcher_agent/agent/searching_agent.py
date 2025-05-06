from autogen import ConversableAgent
from research_searcher_agent.tools.temp_categorization import categorize_research
from research_searcher_agent.tools.query import query_handling, print_papers#, find_basis_paper
from research_searcher_agent.config import LLM_CONFIG

Search_prompt = """
Use the parameters below to search using the API Endpoint described within 'query_handling'.

Parameters: {input}

Once you've found the results, display the results and reply with TERMINATE.
"""


def create_searching_agent() -> ConversableAgent:
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

        3. Find minimum 5 papers.

        4. Use `print_papers()` to display the results to the human.

        5. End your reply with the word TERMINATED once the results have been shown.

        You may ask follow-up questions to clarify incomplete or ambigous input.

        """,
        llm_config=LLM_CONFIG,
    )

    # add the tools to the agent
    agent.register_for_llm(name="temp_categorization", description="Search for Research based on inputs")(categorize_research)
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
    user_proxy.register_for_execution(name="temp_categorization")(categorize_research)
    user_proxy.register_for_execution(name="query_handling")(query_handling)
    user_proxy.register_for_execution(name="print_papers")(print_papers)
    return user_proxy


def main():
    user_proxy = create_user_proxy()
    searching_agent = create_searching_agent()
    # Getting user input
    topic = input("Enter the research title: ")
    year = input("Enter the year: ")
    citations = input("Enter the citations: ")

    # Send user input to the agent as dynamic prompt
    user_input = f"Topic: {topic}, Year: {year}, Citations: {citations}"

    user_proxy.initiate_chat(
        searching_agent, 
        message=Search_prompt.format(input=user_input),
    )

if __name__ == "__main__":
    main()