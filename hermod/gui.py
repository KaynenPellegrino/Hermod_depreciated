# hermod/gui.py

"""
Module: gui.py

Provides a simple graphical user interface for Hermod.
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
from hermod.core.code_generation import generate_code, save_code
from hermod.core.project_generator import generate_project
from hermod.utils.logger import setup_logger
import threading

# Initialize logger
logger = setup_logger()

class HermodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hermod: AI-powered Development Assistant")

        # Command Selection
        self.command_var = tk.StringVar(value="generate_code")
        commands = [("Generate Code", "generate_code"), ("Generate Project", "generate_project")]
        for text, value in commands:
            tk.Radiobutton(root, text=text, variable=self.command_var, value=value, command=self.update_form).pack(anchor=tk.W)

        # Prompt / Specification Input
        self.input_label = tk.Label(root, text="Prompt:")
        self.input_label.pack()
        self.input_text = scrolledtext.ScrolledText(root, width=80, height=10)
        self.input_text.pack()

        # Language Selection
        self.language_label = tk.Label(root, text="Programming Language:")
        self.language_label.pack()
        self.language_entry = tk.Entry(root)
        self.language_entry.insert(0, "Python")
        self.language_entry.pack()

        # Project Name (only for generate_project)
        self.project_name_label = tk.Label(root, text="Project Name:")
        self.project_name_entry = tk.Entry(root)
        self.project_name_entry.insert(0, "GeneratedProject")

        # Generate Button
        self.generate_button = tk.Button(root, text="Execute", command=self.execute_command)
        self.generate_button.pack(pady=10)

        # Output Display
        self.output_label = tk.Label(root, text="Output:")
        self.output_label.pack()
        self.output_text = scrolledtext.ScrolledText(root, width=80, height=10, state='disabled')
        self.output_text.pack()

        self.update_form()

    def update_form(self):
        command = self.command_var.get()
        if command == "generate_project":
            self.project_name_label.pack()
            self.project_name_entry.pack()
        else:
            self.project_name_label.pack_forget()
            self.project_name_entry.pack_forget()

    def execute_command(self):
        command = self.command_var.get()
        input_data = self.input_text.get("1.0", tk.END).strip()
        language = self.language_entry.get().strip()

        if command == "generate_code":
            threading.Thread(target=self.generate_code_thread, args=(input_data, language)).start()
        elif command == "generate_project":
            project_name = self.project_name_entry.get().strip()
            threading.Thread(target=self.generate_project_thread, args=(input_data, project_name, language)).start()

    def generate_code_thread(self, prompt, language):
        self.append_output(f"Generating code for prompt: {prompt} in {language}\n")
        code = generate_code(prompt, language=language)
        if code:
            filename = f'generated_code/{language.lower()}_code.{language.lower()}'
            save_code(code, filename)
            self.append_output(f"Code generated and saved to {filename}\n")
        else:
            self.append_output("Failed to generate code.\n")

    def generate_project_thread(self, specification, project_name, language):
        self.append_output(f"Generating project: {project_name} with specification: {specification} in {language}\n")
        generate_project(specification, project_name=project_name, language=language)
        self.append_output(f"Project {project_name} generated successfully.\n")

    def append_output(self, text):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    gui = HermodGUI(root)
    root.mainloop()
