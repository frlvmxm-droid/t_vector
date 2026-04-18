#!/usr/bin/env python3
"""
Refactoring script: extract СТАДИЯ 1 and СТАДИЯ 4 from run_cluster() into separate methods.

Steps:
1. Add missing fields to ClusterRunState (stamp, use_streaming, use_inc_model)
2. Extract СТАДИЯ 1 as _cluster_worker_stage1()
3. Replace СТАДИЯ 1 block in worker() with checkpoint + call + unpack
4. Extract СТАДИЯ 4 as _cluster_worker_stage4()
5. Replace СТАДИЯ 4 block in worker() with checkpoint + call
"""

import re

SRC = "app_cluster.py"

with open(SRC, "r", encoding="utf-8") as f:
    lines = f.readlines()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_line(text, start=1, end=None):
    """Return 1-based line number of first line containing `text`."""
    lim = end if end else len(lines)
    for i in range(start - 1, lim):
        if text in lines[i]:
            return i + 1
    raise ValueError(f"Could not find: {text!r}")


def de_indent_lines(body_lines, remove_spaces=12):
    """Remove `remove_spaces` leading spaces from each non-empty line."""
    result = []
    for l in body_lines:
        if l.strip() == "":
            result.append("\n")
        elif l.startswith(" " * remove_spaces):
            result.append(l[remove_spaces:])
        else:
            result.append(l)
    return result


def replace_vars_smart(body_lines, replacements):
    """
    Replace standalone variable names with _crs.varname.

    Smart rules:
    - Replace `varname` with `_crs.varname` in most contexts
    - Do NOT replace lambda default-argument parameter names (the left side of `name=value`
      inside a lambda signature)
    - Do NOT replace keyword argument names in function calls
    """
    result = []
    for line in body_lines:
        new_line = _replace_line(line, replacements)
        result.append(new_line)
    return result


def _replace_line(line, replacements):
    """Replace vars in a single line, respecting lambda param names."""
    # Strategy: split line into segments, process each segment carefully
    # For lines containing 'lambda', handle the lambda signature specially

    for var, replacement in replacements:
        if var not in line:
            continue

        # Replace occurrences of `var` but not when they appear as:
        # 1. Lambda parameter names: `lambda foo=...` -- don't replace `foo` there
        # 2. The new_line should change `foo` -> `_crs.foo` in all other positions

        # Use a stateful replacement that tracks whether we're in a lambda signature
        new_line = _smart_replace_var(line, var, replacement)
        line = new_line

    return line


def _smart_replace_var(line, var, replacement):
    """
    Replace `var` with `replacement` in `line`, but skip occurrences that are
    lambda default-argument parameter names (left side of `param=value` in lambda).
    """
    # Find all lambda parameter sections in the line
    # Lambda params are between 'lambda' keyword and the ':'
    # We only skip replacements for names that are PARAMETER names (left side of `=`)
    # not for the values (right side of `=`) or other usages

    # Find positions where `var` is a lambda param name
    # Pattern: 'lambda' ... [, ] var =  (before the lambda colon)
    lambda_param_positions = set()

    # Simple approach: find lambda...colon spans and within them find `var=`
    for m in re.finditer(r'\blambda\b[^:]*:', line):
        lambda_span = m.group(0)
        lambda_start = m.start()
        # Within this lambda signature, find `var=` as a parameter name
        for pm in re.finditer(r'(?<![.\w])' + re.escape(var) + r'\s*=', lambda_span):
            # This is a parameter name, record its absolute position in the line
            abs_pos = lambda_start + pm.start()
            lambda_param_positions.add(abs_pos)

    # Now do the replacement, skipping positions that are lambda param names
    result = []
    pos = 0
    pattern = re.compile(r'(?<![.\w])' + re.escape(var) + r'(?![\w])')

    for m in pattern.finditer(line):
        result.append(line[pos:m.start()])
        if m.start() in lambda_param_positions:
            # Keep as-is (lambda param name)
            result.append(m.group(0))
        else:
            result.append(replacement)
        pos = m.end()

    result.append(line[pos:])
    return "".join(result)


# ---------------------------------------------------------------------------
# STEP 0: Verify key line numbers
# ---------------------------------------------------------------------------
CRS_END_LINE     = find_line("kw_dict: dict = _dc_field(default_factory=dict)")
RUN_CLUSTER_LINE = find_line("    def run_cluster(self):")
STAGE1_START     = find_line("# СТАДИЯ 1: Чтение файлов")
STAGE2_START     = find_line("# СТАДИЯ 2: Векторизация")
STAGE4_START     = find_line('cid_col   = snap["cluster_id_col"]')
EXCEPT_IE_LINE   = find_line("except InterruptedError:", start=STAGE4_START)

print(f"CRS_END_LINE    = {CRS_END_LINE}")
print(f"RUN_CLUSTER_LINE= {RUN_CLUSTER_LINE}")
print(f"STAGE1_START    = {STAGE1_START}")
print(f"STAGE2_START    = {STAGE2_START}")
print(f"STAGE4_START    = {STAGE4_START}")
print(f"EXCEPT_IE_LINE  = {EXCEPT_IE_LINE}")

# ---------------------------------------------------------------------------
# STEP 1: Add missing fields to ClusterRunState
# ---------------------------------------------------------------------------
existing = "".join(lines)
fields_to_add = []
if 'stamp: str = ""' not in existing:
    fields_to_add.append('        stamp: str = ""\n')
if "use_streaming: bool = False" not in existing:
    fields_to_add.append("        use_streaming: bool = False\n")
if "use_inc_model: bool = False" not in existing:
    fields_to_add.append("        use_inc_model: bool = False\n")

if fields_to_add:
    insert_pos = CRS_END_LINE  # insert after kw_dict line
    lines = lines[:insert_pos] + fields_to_add + lines[insert_pos:]
    offset = len(fields_to_add)
    RUN_CLUSTER_LINE += offset
    STAGE1_START += offset
    STAGE2_START += offset
    STAGE4_START += offset
    EXCEPT_IE_LINE += offset
    print(f"Added {len(fields_to_add)} fields to ClusterRunState")
else:
    print("Fields already present in ClusterRunState")

# ---------------------------------------------------------------------------
# STEP 2: Extract СТАДИЯ 1 body
# ---------------------------------------------------------------------------
STAGE1_BODY_LINES = lines[STAGE1_START - 1 : STAGE2_START - 1]
stage1_body_deindented = de_indent_lines(STAGE1_BODY_LINES, remove_spaces=8)

# Variables from CRS that stage1 reads (input) and writes (output)
stage1_crs_vars = [
    ("in_paths", "_crs.in_paths"),
    ("total_rows", "_crs.total_rows"),
    ("start_ts", "_crs.start_ts"),
    ("use_t5", "_crs.use_t5"),
    ("cluster_snap", "_crs.cluster_snap"),
    ("X_all", "_crs.X_all"),
    ("file_data", "_crs.file_data"),
    ("raw_texts_all", "_crs.raw_texts_all"),
    ("done", "_crs.done"),
    ("n_ok", "_crs.n_ok"),
]

stage1_body_with_crs = replace_vars_smart(stage1_body_deindented, stage1_crs_vars)

# Build the method
stage1_method_lines = [
    "    def _cluster_worker_stage1(self, _crs: \"ClusterRunState\", snap: dict) -> None:\n",
    "        \"\"\"СТАДИЯ 1: Чтение файлов, очистка текста, дедупликация.\n",
    "\n",
    "        Reads all input files, builds _crs.X_all / _crs.file_data / _crs.raw_texts_all.\n",
    "        Inputs from _crs: in_paths, total_rows, start_ts, use_t5, cluster_snap.\n",
    "        Outputs to _crs: X_all, file_data, raw_texts_all, done, n_ok.\n",
    "        \"\"\"\n",
]
stage1_method_lines.extend(stage1_body_with_crs)
if stage1_method_lines[-1].strip() != "":
    stage1_method_lines.append("\n")
print(f"Stage1 method: {len(stage1_method_lines)} lines")

# ---------------------------------------------------------------------------
# STEP 3: Replacement block for СТАДИЯ 1 in worker()
# ---------------------------------------------------------------------------
INDENT = "                "  # 16 spaces

stage1_replacement = [
    f"{INDENT}# ═══════════════════════════════════════════════════════════════\n",
    f"{INDENT}# СТАДИЯ 1: Чтение файлов, очистка текста, дедупликация\n",
    f"{INDENT}# ═══════════════════════════════════════════════════════════════\n",
    f"{INDENT}# Checkpoint inputs into _crs\n",
    f"{INDENT}_crs.in_paths = list(in_paths)\n",
    f"{INDENT}_crs.total_rows = total_rows\n",
    f"{INDENT}_crs.start_ts = start_ts\n",
    f"{INDENT}_crs.use_t5 = use_t5\n",
    f"{INDENT}_crs.cluster_snap = cluster_snap\n",
    f"{INDENT}# Run stage 1\n",
    f"{INDENT}self._cluster_worker_stage1(_crs, snap)\n",
    f"{INDENT}# Unpack outputs\n",
    f"{INDENT}X_all = _crs.X_all\n",
    f"{INDENT}file_data = _crs.file_data\n",
    f"{INDENT}raw_texts_all = _crs.raw_texts_all\n",
    f"{INDENT}done = _crs.done\n",
    f"{INDENT}n_ok = _crs.n_ok\n",
    f"\n",
]

# ---------------------------------------------------------------------------
# STEP 4: Extract СТАДИЯ 4 body
# ---------------------------------------------------------------------------
# Find the ═══ separator just before STAGE4_START
sep_line = STAGE4_START - 2
while sep_line > 0 and "═══" not in lines[sep_line - 1]:
    sep_line -= 1
if "═══" not in lines[sep_line - 1]:
    sep_line = STAGE4_START - 2

print(f"STAGE4 section (for extraction) starts at line {sep_line}")

STAGE4_BODY_LINES = lines[sep_line - 1 : EXCEPT_IE_LINE - 1]
stage4_body_deindented = de_indent_lines(STAGE4_BODY_LINES, remove_spaces=8)

stage4_crs_vars = [
    ("labels", "_crs.labels"),
    ("kw_dict", "_crs.kw_dict"),
    ("kw_final", "_crs.kw_final"),
    ("kw", "_crs.kw"),
    ("file_data", "_crs.file_data"),
    ("cluster_name_map", "_crs.cluster_name_map"),
    ("cluster_reason_map", "_crs.cluster_reason_map"),
    ("cluster_quality_map", "_crs.cluster_quality_map"),
    ("llm_feedback_map", "_crs.llm_feedback_map"),
    ("t5_summaries_all", "_crs.t5_summaries_all"),
    ("labels_l1", "_crs.labels_l1"),
    ("noise_labels", "_crs.noise_labels"),
    ("total_rows", "_crs.total_rows"),
    ("use_hdbscan", "_crs.use_hdbscan"),
    ("use_hier", "_crs.use_hier"),
    ("use_lda", "_crs.use_lda"),
    ("K", "_crs.K"),
]

stage4_body_with_crs = replace_vars_smart(stage4_body_deindented, stage4_crs_vars)

stage4_method_lines = [
    "    def _cluster_worker_stage4(\n",
    "        self,\n",
    "        _crs: \"ClusterRunState\",\n",
    "        snap: dict,\n",
    "        t0: float,\n",
    "        ui_prog,\n",
    "        _lifecycle,\n",
    "    ) -> None:\n",
    "        \"\"\"СТАДИЯ 4: Экспорт результатов в XLSX, сводная таблица.\n",
    "\n",
    "        Writes clustered XLSX output files and fires done_ui callback.\n",
    "        Reads from _crs: labels, kw_dict, kw, file_data, cluster_name_map,\n",
    "          cluster_reason_map, cluster_quality_map, llm_feedback_map,\n",
    "          t5_summaries_all, labels_l1, noise_labels, K, total_rows,\n",
    "          use_hdbscan, use_hier, use_lda.\n",
    "        Params: t0 (wall-clock start for elapsed), ui_prog, _lifecycle.\n",
    "        \"\"\"\n",
]
stage4_method_lines.extend(stage4_body_with_crs)
if stage4_method_lines[-1].strip() != "":
    stage4_method_lines.append("\n")
print(f"Stage4 method: {len(stage4_method_lines)} lines")

# ---------------------------------------------------------------------------
# STEP 5: Replacement block for СТАДИЯ 4 in worker()
# ---------------------------------------------------------------------------
stage4_replacement = [
    f"{INDENT}# ═══════════════════════════════════════════════════════════════\n",
    f"{INDENT}# СТАДИЯ 4: Экспорт результатов в XLSX, сводная таблица\n",
    f"{INDENT}# ═══════════════════════════════════════════════════════════════\n",
    f"{INDENT}# Checkpoint state for export\n",
    f"{INDENT}_crs.labels = labels\n",
    f"{INDENT}_crs.kw_dict = kw_dict\n",
    f"{INDENT}_crs.kw = kw\n",
    f"{INDENT}_crs.file_data = file_data\n",
    f"{INDENT}_crs.cluster_name_map = cluster_name_map\n",
    f"{INDENT}_crs.cluster_reason_map = cluster_reason_map\n",
    f"{INDENT}_crs.cluster_quality_map = cluster_quality_map\n",
    f"{INDENT}_crs.llm_feedback_map = llm_feedback_map\n",
    f"{INDENT}_crs.t5_summaries_all = t5_summaries_all\n",
    f"{INDENT}_crs.labels_l1 = labels_l1\n",
    f"{INDENT}_crs.noise_labels = noise_labels\n",
    f"{INDENT}_crs.K = K\n",
    f"{INDENT}self._cluster_worker_stage4(_crs, snap, t0, ui_prog, _lifecycle)\n",
    f"\n",
]

# ---------------------------------------------------------------------------
# STEP 6: Add _crs initialization to worker()
# ---------------------------------------------------------------------------
crs_init_present = any(
    "_crs = " in l and "ClusterRunState" in l
    for l in lines[RUN_CLUSTER_LINE - 1:STAGE1_START - 1]
)
print(f"_crs initialization already present: {crs_init_present}")

done_line = None
for i in range(RUN_CLUSTER_LINE - 1, STAGE1_START - 1):
    if lines[i].strip() == "done = 0":
        done_line = i + 1  # 1-based line number, insert AFTER this line
        break
print(f"'done = 0' is at line {done_line}")

# ---------------------------------------------------------------------------
# Apply all changes in REVERSE order (highest line numbers first)
# ---------------------------------------------------------------------------

# First: STEP 6 - insert _crs init (do this before STAGE1 replacement since it's before)
if not crs_init_present and done_line is not None:
    crs_init = [
        "\n",
        "                _crs = ClusterRunState()\n",
    ]
    lines = lines[:done_line] + crs_init + lines[done_line:]
    shift = len(crs_init)
    STAGE1_START += shift
    STAGE2_START += shift
    STAGE4_START += shift
    EXCEPT_IE_LINE += shift
    sep_line += shift
    print(f"Inserted _crs init after line {done_line}")

# STEP 5: Replace СТАДИЯ 4 in worker() (high line numbers first)
stage4_worker_start = sep_line - 1   # 0-indexed
stage4_worker_end   = EXCEPT_IE_LINE - 1  # 0-indexed exclusive
print(f"Replacing СТАДИЯ 4 in worker(): lines {stage4_worker_start+1}..{stage4_worker_end}")
lines = lines[:stage4_worker_start] + stage4_replacement + lines[stage4_worker_end:]
stage4_shift = len(stage4_replacement) - (stage4_worker_end - stage4_worker_start)

# STEP 3: Replace СТАДИЯ 1 in worker() (lower line numbers, but still in worker)
stage1_worker_start = STAGE1_START - 1   # 0-indexed
stage1_worker_end   = STAGE2_START - 1   # 0-indexed exclusive
print(f"Replacing СТАДИЯ 1 in worker(): lines {stage1_worker_start+1}..{stage1_worker_end}")
lines = lines[:stage1_worker_start] + stage1_replacement + lines[stage1_worker_end:]
stage1_shift = len(stage1_replacement) - (stage1_worker_end - stage1_worker_start)

# STEP 2 + 4: Insert both new methods just before run_cluster()
# Find current position of run_cluster after all modifications
run_cluster_pos = None
for i, line in enumerate(lines):
    if "    def run_cluster(self):" in line:
        run_cluster_pos = i
        break
print(f"Inserting methods before run_cluster at line {run_cluster_pos + 1}")

insertion_methods = []
insertion_methods.extend(stage1_method_lines)
insertion_methods.append("\n")
insertion_methods.extend(stage4_method_lines)
insertion_methods.append("\n")

lines = lines[:run_cluster_pos] + insertion_methods + lines[run_cluster_pos:]
print(f"Inserted {len(insertion_methods)} lines before run_cluster")

# ---------------------------------------------------------------------------
# Write the result
# ---------------------------------------------------------------------------
with open(SRC, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Wrote {len(lines)} lines to {SRC}")

# Quick sanity check: verify the lambda fix
result_text = "".join(lines)
if "lambda _crs." in result_text:
    print("WARNING: Found 'lambda _crs.' in output - possible lambda param replacement issue!")
    # Find and show offending lines
    for i, l in enumerate(lines, 1):
        if "lambda _crs." in l:
            print(f"  Line {i}: {l.rstrip()[:100]}")
else:
    print("Lambda param check: OK (no 'lambda _crs.' found)")
