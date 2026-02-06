from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
)
from langgraph.checkpoint.memory import InMemorySaver


from ai import tools


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    admin_name = request.runtime.context.get("admin_name", "Admin")

    return f"""You are a wingman of {admin_name}. Other users will ask you questions about {admin_name}.
    You can use the tools to answer the questions. Always use the tools when you are not sure about the answer.

    If you don't find any relevant information from the tools, you can try to find close enough information and use it to answer the question.
    eg. If someone asks about {admin_name}'s work in backend dev but you only find information about frontend dev, you can emphasize the frontend dev experience and try to relate it to backend dev.
    """


def get_agent():

    chat_model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.7)

    agent = create_agent(
        model=chat_model,
        tools=tools.get_all(),
        context_schema=tools.Context,
        middleware=[
            system_prompt,
            ToolCallLimitMiddleware(thread_limit=20, run_limit=3),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent
