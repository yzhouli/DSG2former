"""Check source syntax, local imports and excluded private artifacts."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    checked = 0
    ignored = {"data", "outputs", "external", "models", "results", "tmp", "vendor", "__pycache__", ".git"}
    for p in ROOT.rglob("*"):
        relative = p.relative_to(ROOT)
        if ignored.intersection(relative.parts) or not p.is_file():
            continue
        name = str(relative)
        text = p.read_text()
        if re.search(r"/Users/[A-Za-z0-9_.-]+/|/data/data[0-9]/|192[.]168[.][0-9]+[.][0-9]+", text):
            failures.append(f"Private host/home path: {name}")
        if re.search(r"(?:gh[pousr]_)[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
            failures.append(f"Credential-like content: {name}")
        if p.suffix == ".py":
            checked += 1
            tree = ast.parse(text, filename=name)
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    modules = [node.module]
                for module in modules:
                    if module.startswith(("dsg2former.", "scripts.")):
                        target = ROOT / module.replace(".", "/")
                        if not target.with_suffix(".py").is_file() and not target.is_dir():
                            failures.append(f"Missing local dependency: {module} in {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps(dict(passed=True, checked_python_sources=checked)))


if __name__ == "__main__":
    main()
