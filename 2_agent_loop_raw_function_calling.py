from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b"


#  --- Tool (LangChain @tool decorator) ---


@traceable(run_type="tool")
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


@traceable(run_type="tool")
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


# Difference 2: without @tool, we manually define schema for each function
# this is exactly what langchain's @tool decorator generates automatically
# from the function's type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look for the price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount based on the discount tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The price of the product to apply discount to",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# Note - ollama can auto-generate these schemas if you pass the function
# directly as tools (similar to Langchain's @tool decorator):
# tools_for_llm = [get_product_price, apply_discount]
# However, this response your docstrings to follow the Google docstring format

# --------------- Helper: traced ollama call -----------
# Difference 3: without langchain, we must manually trace LLM calls for LangSmith


@traceable(name="Ollama chat call", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    # this is the loop of ReAct framework we are following
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: attribute access .name .args .id instead of dict access []
        tool_name = tool_call.name
        tool_args = tool_call.args
        tool_call_id = tool_call.id

        print(f" >> Tool Call: {tool_name} with args: {tool_args}")

        # Difference 7: direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f" >> Tool Observation: {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: max interations reached without a final answer")
    return None


if __name__ == "__main__":
    print(f" --- LangChain Tool Calling Demo ---")
    result = run_agent("What is the final price of a laptop after a gold discount?")
    print(f" \n FINAL ANSWER: {result}")
