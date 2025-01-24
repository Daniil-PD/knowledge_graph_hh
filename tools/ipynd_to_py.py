import json
from pprint import pprint
import os


if __name__ == "__main__":
    input_file = "graph_obj.ipynb"
    
    new_file_data = ""

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            new_file_data += "\n"
            new_file_data += "".join(cell["source"])

        if cell["cell_type"] == "markdown":
            new_file_data += "\n\n# "
            new_file_data += "".join(cell["source"])

    print(new_file_data)