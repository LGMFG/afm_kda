#!/usr/bin/env python
"""Why do the all-linear students fail MBPP?  Reads lm-eval's samples_mbpp_*.jsonl (no GPU, login node).

    python mbpp_samples.py r1=<samples.jsonl> r2=<...> r3=<...> r4=<...> [teacher=<...>] [--rescue]

Per model, over all 500 problems:
  pass         lm-eval's pass@1
  name ok      the answer defines the function the test asserts call (copied from the prompt)
  [BEGIN]      the answer contains '[BEGIN]': the model started a new example instead of writing
               '[DONE]', so generation ran on until the token limit -> the extra text is part of the
               "code" and usually breaks it
  no stop      the answer does not end with a newline, i.e. it was cut mid-line by the token limit
               instead of ending at '[DONE]'
  parses       the answer is valid Python as returned (what lm-eval executes)
  parses cut   valid Python after cutting at the first '[BEGIN]'
  CRLF         the answer reproduces the '\\r\\n' line endings of the 3 in-context examples
  loop         some line repeats >= 5 times (degenerate repetition)
  --rescue     also EXECUTES the cut answers against the tests (subprocess, 10 s timeout each):
               pass rate if the runaway after '[BEGIN]' were removed.  Runs model-written code --
               harmless snippets in practice, but only do this on a login node / your own account.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile


def expected_names(doc):
    tests = " ".join(doc["test_list"])
    return [n for n in re.findall(r"def\s+(\w+)\s*\(", doc["code"]) if re.search(rf"\b{n}\s*\(", tests)]


def parses(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def run_tests(code, doc, timeout=10):
    prog = "\n".join([doc.get("test_setup_code", ""), code, *doc["test_list"]])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        return subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def analyse(path, rescue):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    c = dict(n=len(rows), pass_=0, name=0, begin=0, nostop=0, parse=0, parsecut=0, crlf=0, loop=0, rescued=0)
    for r in rows:
        doc, resp = r["doc"], r["filtered_resps"][0]
        cut = resp.split("[BEGIN]")[0]
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        c["pass_"] += r["pass_at_1"] > 0
        names = expected_names(doc)
        c["name"] += any(re.search(rf"def\s+{n}\s*\(", resp) for n in names) if names else 0
        c["begin"] += "[BEGIN]" in resp
        c["nostop"] += not resp.endswith("\n")          # cut mid-line = hit the token limit
        c["parse"] += parses(resp)
        c["parsecut"] += parses(cut)
        c["crlf"] += "\r\n" in resp
        c["loop"] += bool(lines) and max(lines.count(l) for l in set(lines)) >= 5
        if rescue:
            c["rescued"] += run_tests(cut, doc)
    return c


def main():
    rescue = "--rescue" in sys.argv
    specs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    cols = [("pass", "pass_"), ("name ok", "name"), ("[BEGIN]", "begin"), ("no stop", "nostop"),
            ("parses", "parse"), ("parses cut", "parsecut"), ("CRLF", "crlf"), ("loop", "loop")]
    if rescue:
        cols.append(("pass if cut", "rescued"))
    print(f"{'model':10}" + "".join(f"{h:>12}" for h, _ in cols) + "    (% of problems)")
    for name, path in specs:
        c = analyse(path, rescue)
        print(f"{name:10}" + "".join(f"{100 * c[k] / c['n']:12.1f}" for _, k in cols) + f"    n={c['n']}")


if __name__ == "__main__":
    main()