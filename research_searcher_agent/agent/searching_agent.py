from autogen import ConversableAgent
from research_searcher_agent.tools.temp_categorization import categorize_research
from research_searcher_agent.tools.query import query_handling, print_papers, find_basis_paper
from research_searcher_agent.config import LLM_CONFIG

Search_prompt = """
Search for the result as best you can. You have access to the tools provided.
Request the parameters of topic, year and citations from the human: {input}
Use the parameters to search using the API Endpoint described within 'query_handling'

Once you've found the results, display the results and reply with TERMINATE.
"""

# Request the parameters of topic, year and citations from the human: {input}


def create_searching_agent() -> ConversableAgent:
    # define the agent
    agent = ConversableAgent(
        name="Research Searcher",
        system_message="You are a helpful AI assistant. "
                    "Request parameters of topic, year and citation from the human"
                      f"Find a research paper on the topic, year and citation count and use it in query_handling ",
        llm_config=LLM_CONFIG,
    )


#"Don't include any other text in your response." "Return 'TERMINATE' when the task is done."
    # add the tools to the agent
    agent.register_for_llm(name="temp_categorization", description="Search for Research based on inputs")(categorize_research)
    agent.register_for_llm(name="query_handling", description="Contains the queries for searching with the API")(query_handling)
    agent.register_for_llm(name="find_basis_paper", description="Contains the queries for searching with the API")(find_basis_paper)
    agent.register_for_llm(name="print_papers", description="Contains the queries for searching with the API")(print_papers)

    return agent


def create_user_proxy():
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    user_proxy.register_for_execution(name="temp_categorization")(categorize_research)
    user_proxy.register_for_execution(name="query_handling")(query_handling)
    user_proxy.register_for_execution(name="find_basis_paper")(find_basis_paper)
    user_proxy.register_for_execution(name="print_papers")(print_papers)
    return user_proxy


def main():
    user_proxy = create_user_proxy()
    searching_agent = create_searching_agent()
    # input = query_handling("world war", "2010", 100)
    user_proxy.initiate_chat(
        searching_agent, 
       # input,
        message=Search_prompt
    )

if __name__ == "__main__":
    main()