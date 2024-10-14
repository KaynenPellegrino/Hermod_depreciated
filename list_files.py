import os

def list_project_files_and_folders_excluding(exclude: list, output_file: str):
    """
    Recursively lists all files and folders in the project directory, excluding specified directories,
    and saves the result with relative paths to an output file.

    :param exclude: List of directory names to exclude from the listing.
    :param output_file: Path to the output file where the result will be saved.
    """
    project_root = os.getcwd()  # Get the current working directory (project root)

    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(project_root):
            # Filter out directories that should be excluded
            dirs[:] = [d for d in dirs if d not in exclude]

            # Get the relative path from the project root
            relative_root = os.path.relpath(root, project_root)

            # Write the current folder path to the file
            if relative_root == '.':
                file.write(f"Folder: {os.path.basename(project_root)}\n")
            else:
                file.write(f"Folder: {relative_root}\n")

            # Write subfolders
            for dir_name in dirs:
                relative_dir_path = os.path.join(relative_root, dir_name)
                file.write(f"  Subfolder: {relative_dir_path}\n")

            # Write all files in the current folder
            for file_name in files:
                relative_file_path = os.path.join(relative_root, file_name)
                file.write(f"  File: {relative_file_path}\n")

    print(f"Project structure saved to: {output_file}")

# Exclude the ".venv" and ".idea" directories
exclude_dirs = ['.venv', '.idea', '.git']

# Define the output file path
output_file_path = r'C:\Users\kayne\Desktop\Sybertnetics\sybertnetics model\hermod work\project_structure_output.txt'

# Call the function to list all files and folders in the project directory, excluding certain directories
list_project_files_and_folders_excluding(exclude_dirs, output_file_path)