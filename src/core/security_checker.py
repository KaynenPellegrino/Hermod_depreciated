import subprocess


def run_security_checks():
    """""\"
Summary of run_security_checks.


Returns
-------
None
""\""""
    print('Running pylint...')
    subprocess.run(['pylint', 'my_project/'])
    print('Running bandit...')
    subprocess.run(['bandit', '-r', 'my_project/'])
