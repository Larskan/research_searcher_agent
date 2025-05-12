import json
from autogen import AssistantAgent
from autogen.oai.client import MistralAIClient
from research_searcher_agent.config import LLM_CONFIG


def load_convo(filepath="full_convo_history.json"):
    with open(filepath, "r", encoding="utf-8") as f:
                history = json.load(f)
    return history
    
def extract_prompt_and_response(history):
    user_msg = next((msg["content"] for msg in history if msg["role"] == "user"), "")

    # Collect all parts of the search agents output and put them together so it reads all parts of it
    agent_parts = []
    for msg in history:
          if msg["role"] in ["assistant", "tool"]:
                #add the content
                if msg.get("content"):
                      agent_parts.append(msg["content"])
                
                # Add tool call content
                tool_calls = msg.get("tool_calls", [])
                if tool_calls is not None: 
                    for call in tool_calls:
                      tool_name = call["function"]["name"]
                      args = call["function"].get("arguments", "")
                      agent_parts.append(f"[TOOL_CALL] {tool_name} {args}")

    agent_response = "\n".join(agent_parts)
    return user_msg.strip(), agent_response.strip()
    
def evaluate_convo_agent(prompt: str, agent_response: str) -> dict:

    llm_config_clean={
            "config_list":[
                {k: v for k, v in LLM_CONFIG["config_list"][0].items() if k != "api_rate_limit"}
            ]
        }
    
    evaluator = AssistantAgent(
    name="search_evaluation_agent",
    system_message="You are an expert evaluator of AI Generated outputs.",
    llm_config=llm_config_clean)

    if isinstance(evaluator.client, MistralAIClient):
        evaluator.client.rate_limit = LLM_CONFIG["config_list"][0].get("api_rate_limit", None)

    critic_prompt = f"""
    You are evaluating an AI Searching Agent.

    Evaluate the response based on these criteria:
    Completeness (1-5): addresses every part of the request.
    Quality (1-5): accurate, clear, and effectively structured.
    Robustness (1-5): handles ambiguities, errors, or nonsensical input well.

    User Prompt: {prompt}
    Agent Response: {agent_response}

    Provide your evaluation as JSON with fields:
    - completeness
    - quality
    - robustness
    - feedback (a brief descriptive explanation including specific examples from the response)
    """
    reply = evaluator.generate_reply([{"role": "user", "content": critic_prompt}])
    try:
        return json.loads(reply["content"])
    except Exception:
        raise ValueError(f"Could not parse input: {reply}")
    
def main():
    history = load_convo()
    prompt, response = extract_prompt_and_response(history)
    evaluation = evaluate_convo_agent(prompt, response)
    print("\n--- Evaluation Result ---\n")
    print(evaluation)

    with open("evaluation_result.json", "w", encoding="utf-8") as f:
              json.dump(evaluation, f, indent=2)
        
if __name__ == "__main__":
        main()

# Note: this works, it is just currently giving a bad result
#