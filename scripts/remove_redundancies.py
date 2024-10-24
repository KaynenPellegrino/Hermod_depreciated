# scripts/remove_redundancies.py

import sys
import logging

from src.utils.logger import get_logger

def remove_redundancies(file_path: str):
    """
    Removes redundant code blocks and unused variables from the specified file.

    Args:
        file_path (str): Path to the Python file to be cleaned.
    """
    logger = get_logger(__name__)
    logger.info(f"Removing redundancies from '{file_path}'.")

    try:
        import ast
        from ast import NodeTransformer, NodeVisitor

        class RedundancyRemover(NodeTransformer):
            """
            AST NodeTransformer to remove redundant code and unused variables.
            """

            def __init__(self):
                self.defined_vars = set()
                self.used_vars = set()

            def visit_FunctionDef(self, node):
                # Collect defined variables
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                self.defined_vars.add(target.id)
                # Visit the function body
                self.generic_visit(node)
                # Remove unused variables
                unused_vars = self.defined_vars - self.used_vars
                if unused_vars:
                    new_body = []
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            targets = stmt.targets
                            if not any(target.id in unused_vars for target in targets if isinstance(target, ast.Name)):
                                new_body.append(stmt)
                        else:
                            new_body.append(stmt)
                    node.body = new_body
                return node

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.used_vars.add(node.id)
                return node

        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        remover = RedundancyRemover()
        tree = remover.visit(tree)
        ast.fix_missing_locations(tree)

        import astor
        cleaned_code = astor.to_source(tree)

        with open(file_path, 'w') as f:
            f.write(cleaned_code)

        logger.debug(f"Redundancies removed from '{file_path}'.")
    except Exception as e:
        logger.error(f"Failed to remove redundancies from '{file_path}': {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_redundancies.py <file_path>")
        sys.exit(1)
    remove_redundancies(sys.argv[1])
