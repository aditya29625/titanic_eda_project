import json
import sys
import io
import os
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Set writable MPLCONFIGDIR inside workspace directory BEFORE importing matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('./.mpl_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook_direct():
    nb_path = "titanic_statistical_eda.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_globals = {}
    exec_locals = exec_globals

    execution_count = 1

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = execution_count
            cell["outputs"] = []

            raw_source = "".join(cell.get("source", []))
            
            def custom_display(*args):
                for arg in args:
                    if hasattr(arg, 'to_string'):
                        print(arg.to_string())
                    else:
                        print(arg)

            exec_globals['display'] = custom_display

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            try:
                clean_lines = []
                for line in raw_source.splitlines():
                    if line.strip().startswith('%') or line.strip().startswith('!'):
                        continue
                    clean_lines.append(line)
                clean_code = "\n".join(clean_lines)

                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(clean_code, exec_globals, exec_locals)
                    plt.close('all')

                out_str = stdout_buf.getvalue()
                if out_str:
                    cell["outputs"].append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": out_str.splitlines(keepends=True)
                    })

                err_str = stderr_buf.getvalue()
                if err_str:
                    cell["outputs"].append({
                        "name": "stderr",
                        "output_type": "stream",
                        "text": err_str.splitlines(keepends=True)
                    })

            except Exception as e:
                err_msg = traceback.format_exc()
                cell["outputs"].append({
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": err_msg.splitlines(keepends=True)
                })
                print(f"Error in cell {execution_count}: {type(e).__name__}: {str(e)}")

            execution_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully executed notebook! Total code cells executed: {execution_count - 1}")

if __name__ == "__main__":
    run_notebook_direct()
