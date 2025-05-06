from autogen import AssistantAgent
from research_searcher_agent.config import LLM_CONFIG
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

def code_executor(text: str) -> str:
    agent = AssistantAgent(
        name="Code Executor",
        system_message="",
        llm_config=LLM_CONFIG,
    )
    reply = agent.generate_reply(
        messages=[
            {"role": "user", "content": f'execute the following: {text}'}
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