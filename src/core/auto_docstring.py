import ast
import astor
import os


def generate_docstring(function_node):
    """Generates a basic docstring for a function or method."""
    param_list = [arg.arg for arg in function_node.args.args]
    docstring = f'"""\nSummary of {function_node.name}.\n\n'
    if param_list:
        docstring += 'Parameters\n'
        docstring += '----------\n'
        for param in param_list:
            docstring += (
                f'{param} : type\n    Description of parameter `{param}`.\n')
    docstring += '\nReturns\n-------\nNone\n"""'
    return docstring


def add_docstrings(file_path):
    """Process a Python file to add or update docstrings."""
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())


    class FunctionVisitor(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            if not ast.get_docstring(node):
                docstring = generate_docstring(node)
                node.body.insert(0, ast.Expr(value=ast.Str(s=docstring)))
                print(f'Adding docstring to {node.name} in {file_path}')
    visitor = FunctionVisitor()
    visitor.visit(tree)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(astor.to_source(tree))


def process_directory(directory_path):
    """Recursively process all Python files in the given directory."""
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f'Processing {file_path}...')
                add_docstrings(file_path)


process_directory('C:/Users/kayne/PycharmProjects/Hermod/src')
