import ast
import difflib
import sys

def get_function_source(filepath, func_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Find start and end lines in original source
            lines = content.splitlines()
            # Ast node doesn't have easy end line in older python versions, but we can find it via children 
            # or just find the line range in recent Python (3.8+)
            if hasattr(node, 'end_lineno'):
                return lines[node.lineno-1:node.end_lineno]
            else:
                # Fallback: just give the first 50 lines starting at lineno
                return lines[node.lineno-1:node.lineno+50]
    return None

def compare(func_name):
    src_ads = get_function_source('app.py', func_name)
    src_prod = get_function_source('scratch/prod_app.py', func_name)
    
    if src_ads is None:
        print(f"Function {func_name} not found in app.py")
        return
    if src_prod is None:
        print(f"Function {func_name} not found in prod_app.py")
        return
    
    diff = difflib.unified_diff(src_prod, src_ads, fromfile='prod', tofile='ads', lineterm='')
    diff_list = list(diff)
    
    outpath = f"scratch/diff_{func_name}.txt"
    with open(outpath, 'w', encoding='utf-8') as f_out:
        if not diff_list:
            f_out.write(f"--- {func_name} is IDENTICAL ---\n")
        else:
            f_out.write(f"--- Diff for {func_name} ---\n")
            for line in diff_list:
                f_out.write(line + "\n")
    print(f"Wrote diff for {func_name} to {outpath}")

if __name__ == '__main__':
    funcs = ['_analyze_library', '_dashboard_html', 'analyze_vibe', 'run_pipeline', 'dashboard_view']
    for f in funcs:
        compare(f)
