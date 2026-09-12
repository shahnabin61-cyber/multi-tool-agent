import ast
import operator
import io
import contextlib
from ddgs import DDGS


ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {node}")


def calculator(expression: str) -> str:
    """Evaluate a math expression safely (no arbitrary code execution)."""
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error: {e}"


def execute_code(code: str) -> str:
    """Run restricted Python code and return printed output."""
    output_buffer = io.StringIO()
    try:
        safe_globals = {"__builtins__": {
            "print": print, "range": range, "len": len,
            "sum": sum, "min": min, "max": max,
            "sorted": sorted, "list": list, "dict": dict,
            "str": str, "int": int, "float": float
        }}
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        result = output_buffer.getvalue()
        return result if result else "Code executed successfully (no output printed)"
    except Exception as e:
        return f"Error: {e}"