import re

TOOLS_DESCRIPTION = """
You have access to these tools:
1. calculator(expression) - evaluates a math expression, e.g. calculator("47 * 892")
2. web_search(query) - searches the web, e.g. web_search("population of Nepal")
3. execute_code(code) - runs Python code and returns printed output, e.g. execute_code("print(sum([1,2,3]))")

Respond in this exact format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<input>)

Once you have enough information to answer, respond instead with:
Thought: <your reasoning>
Final Answer: <your answer to the user>
"""


def run_tool(action_text, calculator, web_search, execute_code):
    match = re.match(r'(\w+)\((.*)\)', action_text.strip(), re.DOTALL)
    if not match:
        return "Error: could not parse action"
    tool_name, tool_input = match.group(1), match.group(2).strip().strip('"').strip("'")

    if tool_name == "calculator":
        return calculator(tool_input)
    elif tool_name == "web_search":
        return web_search(tool_input)
    elif tool_name == "execute_code":
        return execute_code(tool_input)
    else:
        return f"Error: unknown tool {tool_name}"


def run_agent(question, gemini_client, tools, model="gemini-3.6-flash", max_steps=5):
    """
    tools: dict with keys 'calculator', 'web_search', 'execute_code' mapping to callables
    """
    history = f"{TOOLS_DESCRIPTION}\n\nQuestion: {question}\n"

    for step in range(max_steps):
        response = gemini_client.models.generate_content(model=model, contents=history)
        text = response.text
        print(f"--- Step {step+1} ---")
        print(text)

        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        action_match = re.search(r'Action:\s*(.+)', text)
        if action_match:
            action_text = action_match.group(1).strip()
            observation = run_tool(
                action_text, tools['calculator'], tools['web_search'], tools['execute_code']
            )
            print(f"Observation: {observation}\n")
            history += f"{text}\nObservation: {observation}\n"
        else:
            return "Agent did not produce a valid action or final answer."

    return "Max steps reached without a final answer."