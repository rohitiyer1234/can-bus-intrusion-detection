import os

def generate_tree(dir_path, prefix=""):
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return []
    
    # Strictly exclude .venv and __pycache__ folders
    entries = [e for e in entries if e not in ('.venv', '__pycache__')]
    
    lines = []
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (i == len(entries) - 1)
        connector = "+--- " if is_last else "|--- "
        
        lines.append(f"{prefix}{connector}{entry}")
        
        if os.path.isdir(path):
            next_prefix = prefix + ("    " if is_last else "|   ")
            lines.extend(generate_tree(path, next_prefix))
    return lines

if __name__ == "__main__":
    current_dir = os.getcwd()
    header = [
        "Folder PATH listing",
        f"Volume serial number is clean",
        f"{current_dir}\n."
    ]
    tree_structure = header + generate_tree(current_dir)
    
    with open("structure.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tree_structure))
    print("Successfully wrote clean structure.txt!")
