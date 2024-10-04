import tkinter as tk
from threading import Thread


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
        self.text_area = tk.Text(self.root, height=20, width=80)
        self.text_area.pack()

    def update_status(self, message):
        """""\"
Summary of update_status.

Parameters
----------
self : type
    Description of parameter `self`.
message : type
    Description of parameter `message`.

Returns
-------
None
""\""""
        self.text_area.insert(tk.END, message + '\n')

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


def run_hermod_actions(gui, hermod_ai):
    """""\"
Summary of run_hermod_actions.

Parameters
----------
gui : type
    Description of parameter `gui`.
hermod_ai : type
    Description of parameter `hermod_ai`.

Returns
-------
None
""\""""
    while True:
        gui.update_status('Running tests...')
        hermod_ai.run_tests()
        gui.update_status('Tests complete.')
