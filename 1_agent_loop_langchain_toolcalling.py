from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b"


#  --- Tool (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look for the price of a product in the catalog"""
    print(f" >> Executing get_product_price for product: {product}")
    prices = {
        "mouse": 25.99,
        "keyboard": 45.99,
        "monitor": 199.99,
        "webcam": 35.99,
        "laptop": 899.99,
    }
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply discount based on the discount tier.
    discount_tier: 'bronze', 'silver', 'gold'
    """
    print(f" >> Applying discount {discount_tier} to price {price}")
    discount_percentage = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly\n"
                "1. Never guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply discount AFTER you have received"
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # this is the loop of ReAct framework we are following
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" >> Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool name: {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f" >> Tool Observation: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: max interations reached without a final answer")
    return None


if __name__ == "__main__":
    print(f" --- LangChain Tool Calling Demo ---")
    result = run_agent("What is the final price of a laptop after a gold discount?")
    print(f" \n FINAL ANSWER: {result}")
