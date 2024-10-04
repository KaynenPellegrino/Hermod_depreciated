import os
import shutil


class VersionControl:

    def save_project(self, code, project_name):
        """""\"
Summary of save_project.

Parameters
----------
self : type
    Description of parameter `self`.
code : type
    Description of parameter `code`.
project_name : type
    Description of parameter `project_name`.

Returns
-------
None
""\""""
        project_dir = f'./projects/{project_name}'
        version_dir = f'{project_dir}/versions'
        latest_file = f'{version_dir}/main.py'
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        if os.path.exists(latest_file):
            version_num = len(os.listdir(version_dir))
            shutil.copy(latest_file, f'{version_dir}/main_v{version_num}.py')
        with open(latest_file, 'w') as project_file:
            project_file.write(code)

    def rollback_project(self, project_name):
        """""\"
Summary of rollback_project.

Parameters
----------
self : type
    Description of parameter `self`.
project_name : type
    Description of parameter `project_name`.

Returns
-------
None
""\""""
        version_dir = f'./projects/{project_name}/versions'
        if not os.path.exists(version_dir) or len(os.listdir(version_dir)
            ) == 0:
            print('No previous versions available to rollback to.')
            return None
        versions = sorted(os.listdir(version_dir), reverse=True)
        latest_version = versions[0]
        with open(f'{version_dir}/{latest_version}', 'r') as version_file:
            return version_file.read()
