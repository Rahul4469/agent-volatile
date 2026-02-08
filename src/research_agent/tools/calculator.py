import ast
import math
import operator
from typing import Any

from langchain.core.tools import tool

from research_agent.core.exceptions import ToolError

# SAFE EXPRESSION EVALUATOR
# We implement a SAFE expression evaluator that:
# 1. Only allows mathematical operations
# 2. No arbitrary function calls
# 3. No variable assignments
# 4. No imports or __builtins__
#
# This prevents security vulnerabilities like:
#   calculator("__import__('os').system('rm -rf /')")

# Allowed operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # Unary minus
    ast.UAdd: operator.pos,  # Unary plus
}

# Allowed math functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
}

# Allowed constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

class SafeEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> Any:
        """Visit the root Expression node."""
        return self.visit(node.body)
    
    def visit_Num(self, node: ast.Num) -> float:
        """Visit a number literal (Python 3.7 style)."""
        return float(node.n)
    
    def visit_Constant(self, node: ast.Constant) -> float:
        """Visit a constant (Python 3.8+ style)."""
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def visit_Name(self, node: ast.Name) -> float:
        """Visit a name (variable or constant)."""
        name = node.id.lower()
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown variable: {node.id}. Available: {list(SAFE_CONSTANTS.keys())}")
    
    def visit_BinOp(self, node: ast.BinOp) -> float:
        """Visit a binary operation (a + b, a * b, etc.)."""
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        return SAFE_OPERATORS[op_type](left, right)
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """Visit a unary operation (-a, +a)."""
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        
        operand = self.visit(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    
    def visit_Call(self, node: ast.Call) -> float:
        """Visit a function call (sqrt(x), sin(x), etc.)."""
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        
        func_name = node.func.id.lower()
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(
                f"Unknown function: {func_name}. "
                f"Available: {list(SAFE_FUNCTIONS.keys())}"
            )
        
        # Evaluate arguments
        args = [self.visit(arg) for arg in node.args]
        
        return SAFE_FUNCTIONS[func_name](*args)
    
    def visit_List(self, node: ast.List) -> list:
        """Visit a list (for functions like max([1,2,3]))."""
        return [self.visit(elt) for elt in node.elts]
    
    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        """Visit a tuple."""
        return tuple(self.visit(elt) for elt in node.elts)
    
    def generic_visit(self, node: ast.AST) -> Any:
        """Handle unknown node types (security: reject them)."""
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    Args:
        expression: Mathematical expression string
    Returns:
        Evaluation result
    Raises:
        ValueError: If expression is invalid or unsafe
    """    
    try:
        # Parse expression into AST
        tree = ast.parse(expression, mode='eval')
        # Evaluate safely
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)
        
        return float(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")

# CALCULATOR TOOL

@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Use this tool when you need to perform calculations. The tool supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Math functions: sqrt, sin, cos, tan, log, log10, exp, floor, ceil
    - Aggregates: min, max, sum, abs, round
    - Constants: pi, e, tau
    
    Args:
        expression: Mathematical expression to evaluate.
                   Examples: "2 + 2", "sqrt(16)", "sin(pi/2)", "max(1, 2, 3)"
    
    Returns:
        The result as a string, or an error message.
    
    Examples:
        calculator("2 + 2")           → "4.0"
        calculator("sqrt(16) * 2")    → "8.0"
        calculator("sin(pi / 2)")     → "1.0"
        calculator("15 * 847 / 100")  → "127.05"  (15% of 847)
    """
    try:
        result = safe_eval(expression) 
        
        # Format nicely 
        if result == int(result):
            return str(int(result))
        return str(round(result, 10)) # Avoid floating point noise 
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Calcultion failed: {e}"
    

# ALTERNATIVE: CLASS BASED TOOL ---------------------------------------------------
# Useful if you need configuration or dependency injection.  

from research_agent.tools.base import ResearchTool, CalculatorInput

class CalculatorToolClass(ResearchTool):
    """
    Calculator tool implemented as a class.
    
    This shows the class-based approach for tools that need:
    - Configuration
    - State
    - Dependency injection
    
    For simple tools like calculator, the @tool decorator is sufficient.
    But for complex tools (API clients, etc.), classes are better.
    """
    name: str = "calculator"
    description: str = """Evaluate mathematical expressions.
        Supports: +, -, *, /, //, %, ** and functions like sqrt, sin, cos, log.
        Constants: pi, e, tau
        """
    args_schema: type = CalculatorInput 
    
    # Configuration
    precision: int = 10
    
    async def _execeute(self, expression: str) -> str:
        """Execute the calculator."""
        try:
            result = safe_eval(expression)
            
            if result == int(result):
                return str(int(result))
            return str(round(result, self.precision))
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            raise ToolError(
                message=str(e),
                tool_name=self.name,
                tool_args={"expression": expression},
                cause=e,
            )       

# EXPORTS

__all__ = [
    "calculator_tool",
    "CalculatorToolClass",
    "safe_eval",   
]