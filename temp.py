import os
from pathlib import Path

def print_tree(directory, prefix="", max_depth=3, current_depth=0, exclude_dirs=None):
    if current_depth > max_depth:
        return
    
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', 'data', '.pytest_cache', 'node_modules'}
    
    directory = Path(directory)
    items = sorted([item for item in directory.iterdir() 
                   if item.name not in exclude_dirs])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        
        if item.is_dir():
            print(f"{prefix}{current_prefix}{item.name}/")
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(item, next_prefix, max_depth, current_depth + 1, exclude_dirs)
        else:
            print(f"{prefix}{current_prefix}{item.name}")

# Run it - excludes data folder by default
print_tree(".", max_depth=3)