# src/modules/code_generation/staging.py

from language_models.code_gen_model import CodeGenModelInterface, OpenAIModel, MockCodeGenModel
from language_models.syntax_checker import SyntaxCheckerInterface, PythonSyntaxChecker, JavaScriptSyntaxChecker, JavaSyntaxChecker
from ai_project_recommender import TrendAnalyzer, AIProjectRecommender
from code_generator import CodeGenerator
from doc_updater import DocUpdater
from documentation_generator import DocumentationGenerator, ProjectManagerInterface, TemplateManagerInterface
from project_auto_optimizer import ProjectAutoOptimizer
from project_manager import ProjectManager, VersionControlInterface
from template_manager import TemplateManager, TemplateManagerInterface

# Expose these classes and interfaces to make them easily importable
__all__ = [
    "CodeGenModelInterface",
    "OpenAIModel",
    "MockCodeGenModel",
    "SyntaxCheckerInterface",
    "PythonSyntaxChecker",
    "JavaScriptSyntaxChecker",
    "JavaSyntaxChecker",
    "TrendAnalyzer",
    "AIProjectRecommender",
    "CodeGenerator",
    "TemplateManager",
    "TemplateManagerInterface",
    "ProjectManager",
    "ProjectManagerInterface",
    "DocUpdater",
    "DocumentationGenerator",
    "ProjectAutoOptimizer",
    "VersionControlInterface",
]
