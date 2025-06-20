from typing import Dict, Any
from uuid import uuid4

from GridCalEngine.Utils.Symbolic.symbolic import Var, _expr_to_dict


class Equation:
    """
    Represents a symbolic equation of the form: output = expression
    """

    def __init__(self, output: Var, expression: Any):
        """
        :param output: The left-hand-side variable (Var)
        :param expression: The symbolic expression defining the right-hand side (built using your custom symbolic framework)
        """
        self.uid: int = uuid4().int
        self.output = output
        self.expression = expression

    def __str__(self):
        return f"{self.output} = {self.expression}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the equation to a dictionary
        You need a way to serialize both Var and expression.
        """
        return {
            "idtag": self.uid,
            "output": self.output.uid,
            "eq": _expr_to_dict(self.expression)
        }

    # def from_dict(self, data: Dict[str, Any], var_dict: Dict[int, Var]):
    #     """
    #     Deserialize the equation from a dictionary
    #     :param data: Dictionary with keys 'idtag', 'output', 'eq'
    #     :param var_dict: Dictionary of Var objects keyed by idtag
    #     """
    #     self.idtag = data["idtag"]
    #     self.output = var_dict.get(data["output"])
    #     if self.output is None:
    #         raise ValueError(f"Output variable with id {data['output']} not found in var_dict")
    #
    #
    #     self.expression = Expression.parse(data["eq"])
