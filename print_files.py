import os

def save_files_and_contents(folder_path, output_directory):
    """
    Saves all file names and their contents from a specified folder into an output file.

    Args:
        folder_path (str): The path of the folder to list files from.
        output_directory (str): The directory where the output file will be saved.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Extract the folder name from the provided path
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_file = os.path.join(output_directory, f"Printed Files for {folder_name}.txt")

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"Files and their contents in '{folder_name}':\n\n")
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                out_file.write(f"--- {relative_path} ---\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        out_file.write(f.read())
                except Exception as e:
                    out_file.write(f"Could not read file {file}: {e}\n")
                out_file.write("\n\n")  # Separate files with a blank line

    print(f"File contents saved to '{output_file}'")

# Specify the folder path and output directory
folder_path = ("src/modules/advanced_security")  # Example folder
output_directory = r"C:\Users\kayne\Desktop\Sybertnetics\sybertnetics model\hermod work"  # Output folder
save_files_and_contents(folder_path, output_directory)
