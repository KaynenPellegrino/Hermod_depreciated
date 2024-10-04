import tkinter as tk
from tkinter import scrolledtext


class HermodGUI:

    def __init__(self, hermod_ai):
        """""\"
Summary of __init__.

Parameters
----------
self : type
    Description of parameter `self`.
hermod_ai : type
    Description of parameter `hermod_ai`.

Returns
-------
None
""\""""
        self.hermod_ai = hermod_ai
        self.root = tk.Tk()
        self.root.title('Hermod AI')
        self.project_name_label = tk.Label(self.root, text='Project Name:')
        self.project_name_label.pack(pady=5)
        self.project_name_entry = tk.Entry(self.root, width=50)
        self.project_name_entry.pack(pady=5)
        self.modification_label = tk.Label(self.root, text=
            'Modification Prompt:')
        self.modification_label.pack(pady=5)
        self.modification_entry = tk.Entry(self.root, width=50)
        self.modification_entry.pack(pady=5)
        self.text_area = scrolledtext.ScrolledText(self.root, width=80,
            height=20)
        self.text_area.pack(pady=10)
        self.generate_button = tk.Button(self.root, text=
            'Generate/Update Code', command=self.generate_code)
        self.generate_button.pack(pady=10)
        self.rollback_button = tk.Button(self.root, text='Rollback Code',
            command=self.rollback_code)
        self.rollback_button.pack(pady=10)
        self.error_label = tk.Label(self.root, text='')
        self.error_label.pack(pady=5)

    def generate_code(self):
        """""\"
Summary of generate_code.

Parameters
----------
self : type
    Description of parameter `self`.

Returns
-------
None
""\""""
        project_name = self.project_name_entry.get()
        code, errors = self.hermod_ai.update_code(project_name)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.INSERT, code)
        if errors:
            self.error_label.config(text=f'Errors: {errors}')
        else:
            self.error_label.config(text='No errors found.')

    def rollback_code(self):
        """""\"
Summary of rollback_code.

Parameters
----------
self : type
    Description of parameter `self`.

Returns
-------
None
""\""""
        project_name = self.project_name_entry.get()
        rollback_code = self.hermod_ai.rollback_code(project_name)
        self.text_area.delete(1.0, tk.END)
        if rollback_code:
            self.text_area.insert(tk.INSERT, rollback_code)

    def run(self):
        """""\"
Summary of run.

Parameters
----------
self : type
    Description of parameter `self`.

Returns
-------
None
""\""""
        self.root.mainloop()
