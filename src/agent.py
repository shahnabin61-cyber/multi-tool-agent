import re

TOOLS_DESCRIPTION = """
You have access to these tools:
1. calculator(expression) - evaluates a math expression, e.g. calculator("47 * 892")
2. web_search(query) - searches the web for current/general information
3. execute_code(code) - runs Python code and returns printed output
4. retrieve_documents(query) - searches indexed documents (e.g. a legal/constitutional
   text) for specific provisions

Use retrieve_documents for questions about the indexed document's content.
Use web_search for general current information.
Use calculator for arithmetic.

Respond in this exact format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<input>)

Once you have enough information to answer, respond instead with:
Thought: <your reasoning>
Final Answer: <your answer to the user>
"""


def run_tool(action_text, tools):
    match = re.match(r'(\w+)\((.*)\)', action_text.strip(), re.DOTALL)
    if not match:
        return "Error: could not parse action"
    tool_name, tool_input = match.group(1), match.group(2).strip().strip('"').strip("'")

    if tool_name in tools:
        return tools[tool_name](tool_input)
    return f"Error: unknown tool {tool_name}"


def run_agent(question, gemini_client, tools, model="gemini-3.6-flash", max_steps=5):
    """tools: dict mapping tool name -> callable(str) -> str"""
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
            observation = run_tool(action_text, tools)
            print(f"Observation: {observation}\n")
            history += f"{text}\nObservation: {observation}\n"
        else:
            return "Agent did not produce a valid action or final answer."

    return "Max steps reached without a final answer."