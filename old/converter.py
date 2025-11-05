import json
import re

def parse_md_to_cells(content):
    cells = []
    # Split by '---\n'
    parts = content.split('---\n')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if starts with ***Cell
        if part.startswith('***Cell'):
            # Extract type
            if 'Markdown***' in part:
                cell_type = 'markdown'
                # Find the content between ````markdown and ````
                match = re.search(r'````markdown\n(.*?)\n````', part, re.DOTALL)
                if match:
                    source_text = match.group(1)
                    source = source_text.split('\n')
                    source = [line + '\n' for line in source]
                    cells.append({
                        "cell_type": "markdown",
                        "source": source,
                        "metadata": {}
                    })
            elif 'Code***' in part:
                cell_type = 'code'
                # Between ````python and ````
                match = re.search(r'````python\n(.*?)\n````', part, re.DOTALL)
                if match:
                    source_text = match.group(1)
                    source = source_text.split('\n')
                    source = [line + '\n' for line in source]
                    cells.append({
                        "cell_type": "code",
                        "source": source,
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None
                    })
    return cells

def convert_md_to_ipynb(md_file, ipynb_file):
    with open(md_file, 'r') as f:
        content = f.read()
    cells = parse_md_to_cells(content)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    import sys
    md_file = sys.argv[1]
    ipynb_file = sys.argv[2]
    convert_md_to_ipynb(md_file, ipynb_file)