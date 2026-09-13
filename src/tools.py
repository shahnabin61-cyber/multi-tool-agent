import ast
import operator
import io
import contextlib
from ddgs import DDGS
from sentence_transformers import CrossEncoder


ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
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
        return str(_safe_eval(tree.body))
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
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "sorted": sorted, "list": list,
            "dict": dict, "str": str, "int": int, "float": float
        }}
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        result = output_buffer.getvalue()
        return result if result else "Code executed successfully (no output printed)"
    except Exception as e:
        return f"Error: {e}"


class DocumentRetriever:
    """Hybrid retrieval (embeddings + BM25 candidates) with cross-encoder re-ranking.

    Naive score blending and Reciprocal Rank Fusion were both tried and found
    insufficient when BM25 and embeddings independently agree on a wrong answer
    (correlated error) rather than making different mistakes. Cross-encoder
    re-ranking fixes this by scoring query-document pairs jointly.
    """

    def __init__(self, chunks, embed_model, index):
        self.chunks = chunks
        self.embed_model = embed_model
        self.index = index
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, initial_k: int = 15, final_k: int = 3) -> str:
        try:
            query_embedding = self.embed_model.encode([query])
            distances, indices = self.index.search(
                query_embedding.astype('float32'), k=initial_k
            )
            candidates = [self.chunks[i] for i in indices[0]]

            pairs = [[query, doc] for doc in candidates]
            scores = self.cross_encoder.predict(pairs)

            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return "\n\n".join([doc for doc, score in reranked[:final_k]])
        except Exception as e:
            return f"Error: {e}"