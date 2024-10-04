import psutil


def check_health():
    """""\"
Summary of check_health.


Returns
-------
None
""\""""
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return {'cpu': cpu_usage, 'memory': memory_usage}


def reinstall_dependencies():
    """""\"
Summary of reinstall_dependencies.


Returns
-------
None
""\""""
    import subprocess
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
