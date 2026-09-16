import os
import json
from pathlib import Path

# Cấu hình bộ lọc
IGNORE_DIRS = {'.git', 'node_modules', 'dist', 'build', '.vscode', 'public', 'uploads', '__pycache__', 'venv', '.ipynb_checkpoints'}
IGNORE_FILES = {'project_crawler.py', 'COMBINE_FOR_AI_Project.txt', '.env'}
ALLOWED_EXTENSIONS = {'.py', '.ipynb', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.json', '.md'}

def extract_code_from_ipynb(filepath):
    """Trích xuất code thuần từ file ipynb"""
    code_content = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            for cell in notebook.get('cells', []):
                if cell['cell_type'] == 'code':
                    code_content += "".join(cell['source']) + "\n\n"
    except Exception as e:
        return f"# Lỗi đọc file notebook: {e}"
    return code_content

def process_directory(dir_path, prefix=''):
    global tree_structure, file_contents
    items = sorted([item for item in os.listdir(dir_path) if item not in IGNORE_DIRS and item not in IGNORE_FILES])
    
    for index, item in enumerate(items):
        full_path = os.path.join(dir_path, item)
        is_last = (index == len(items) - 1)
        branch = '└── ' if is_last else '├── '
        
        if os.path.isdir(full_path):
            tree_structure += f"{prefix}{branch}{item}/\n"
            process_directory(full_path, prefix + ('    ' if is_last else '│   '))
        else:
            tree_structure += f"{prefix}{branch}{item}\n"
            ext = Path(item).suffix
            
            if ext in ALLOWED_EXTENSIONS:
                content = ""
                if ext == '.ipynb':
                    content = extract_code_from_ipynb(full_path)
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                relative_path = os.path.relpath(full_path, os.getcwd()).replace('\\', '/')
                file_contents += f"\n\n--- BẮT ĐẦU FILE: {relative_path} ---\n\n"
                file_contents += content
                file_contents += f"\n\n--- KẾT THÚC FILE: {relative_path} ---\n"

# Thực thi
tree_structure = "CẤU TRÚC THƯ MỤC:\n"
file_contents = "\nCHI TIẾT MÃ NGUỒN:\n"
process_directory(os.getcwd())
with open('COMBINE_FOR_AI_Project.txt', 'w', encoding='utf-8') as f:
    f.write(tree_structure + file_contents)
print("✅ Đã tạo COMBINE_FOR_AI_Project.txt (Đã xử lý file .ipynb)")