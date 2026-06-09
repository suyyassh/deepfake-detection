from pathlib import Path

TARGET_DIRECTORY = "/home/suyash/deepfake-detection-v2/data/raw"
OUTPUT_FILENAME = "directory_structure.txt"

def write_tree(directory, f, prefix="", limit=3):
    path = Path(directory)
    
    # print root directory name on the first call
    if prefix == "":
        print(path.name or ".", file=f)
        
    try:
        # sort entries alphabetically
        entries = sorted(list(path.iterdir()), key=lambda x: x.name)
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]", file=f)
        return

    for i, entry in enumerate(entries):
        # truncate if we hit the limit
        if i >= limit:
            print(f"{prefix}└── ... ({len(entries) - limit} more entries hidden)", file=f)
            break

        # determine the correct ASCII connector
        is_last = (i == len(entries) - 1) 
        if i == limit - 1 and len(entries) > limit:
            is_last = False # the "more entries" line will be the actual last line

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}", file=f)
        
        # recurse into directories
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            write_tree(entry, f, prefix + extension, limit)

if __name__ == "__main__":
    target_path = Path(TARGET_DIRECTORY)
    
    # check if the provided directory actually exists
    if not target_path.exists() or not target_path.is_dir():
        print(f"Error: The directory '{TARGET_DIRECTORY}' does not exist.")
    else:
        # get the directory where THIS script is located
        script_dir = Path(__file__).parent.resolve()
        
        # combine the script's directory with the output filename
        output_path = script_dir / OUTPUT_FILENAME

        # open the file and write the tree into it
        with open(output_path, "w", encoding="utf-8") as f:
            write_tree(TARGET_DIRECTORY, f, limit=3)
            
        print("Success!")