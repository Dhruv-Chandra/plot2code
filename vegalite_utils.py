import ast
import hashlib
import json
import re, os
from copy import deepcopy
from pathlib import Path
import numpy as np

max_workers = max(1, (os.cpu_count() or 4) - 2)
SUPPORTED_AGGREGATES = {"mean", "sum", "count", "median", "min", "max"}
SUPPORTED_PLOT_SUBSET = {
    "line",
    "scatter",
    "bar",
    "histogram",
    "heatmap",
    "boxplot",
    "area",
    "multi-line",
    "grouped-bar",
    "pie",
    "step",
}
VEGALITE_SPEC_KEYS = (
    "vega_spec",
    "vega_lite_spec",
    "vegalite_spec",
    "vega_lite_json",
    "vegalite_json",
    "spec",
)

VEGA_LITE_INSTRUCTION = (
    "You are an expert data visualization engineer.\n"
    "Return ONLY a valid Vega-Lite JSON specification that recreates the given plot.\n\n"
    "STRICT REQUIREMENTS:\n"
    "1. Return a single JSON object, not Python code and not markdown.\n"
    "2. Include inline data in data.values whenever the values are visually inferable.\n"
    "3. Use ONLY this supported plot subset: line, scatter, bar, histogram, heatmap, boxplot, area, multi-line, grouped bar.\n"
    "4. Use Vega-Lite mark, encoding, scale, axis, title, transform, and layer fields as needed.\n"
    "5. Match the visible plot type, colors, markers, labels, scales, legends, annotations, and layout.\n"
    "6. Do NOT add extra styling, grids, titles, legends, annotations, or subplots that are not visible.\n"
    "7. Prefer simple Vega-Lite v5/v6-compatible JSON.\n\n"
    "Return ONLY raw JSON with no markdown or explanation."
)

UNIFIED_INSTRUCTION = (
    "You are an expert Python data visualization engineer.\n"
    "Return ONLY executable Python code that recreates the given plot.\n\n"
    "STRICT REQUIREMENTS:\n"
    "1. Use matplotlib.pyplot as plt and numpy as np when needed.\n"
    "2. Define all data explicitly with no placeholders.\n"
    "3. Match the visible plot type, colors, markers, labels, scales, legends, annotations, and layout.\n"
    "4. Do NOT add extra styling, grids, titles, legends, annotations, or subplots that are not visible.\n"
    "5. Code MUST run without errors.\n"
    "6. End with plt.show().\n\n"
    "Return ONLY raw Python code with no markdown or explanation."
)

VEGA_SPEC_DIR = Path("train_data/vega_specs")

def _mark_type(spec):
    mark = spec.get("mark")
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        return mark.get("type")
    return None


def _unwrap_title(title):
    if isinstance(title, dict):
        text = title.get("text", "")
        if isinstance(text, list):
            return " ".join(str(part) for part in text)
        return text
    if isinstance(title, list):
        return " ".join(str(part) for part in title)
    return title


def _enc_label(enc_channel):
    if not isinstance(enc_channel, dict):
        return ""
    if "title" in enc_channel:
        return _unwrap_title(enc_channel["title"])
    if "aggregate" in enc_channel and enc_channel.get("field"):
        return f"{enc_channel['aggregate']}({enc_channel['field']})"
    if enc_channel.get("aggregate") == "count":
        return "count()"
    return enc_channel.get("field", "")


def _has_facet(spec):
    enc = spec.get("encoding", {})
    return "column" in enc or "row" in enc or "facet" in spec


def _channel(spec, name):
    channel = spec.get("encoding", {}).get(name, {})
    if isinstance(channel, list):
        return channel[0] if channel else {}
    return channel if isinstance(channel, dict) else {}


def _field(spec, name):
    return _channel(spec, name).get("field")


def _value(spec, name):
    channel = _channel(spec, name)
    if "value" in channel:
        return channel["value"]
    mark = spec.get("mark")
    if isinstance(mark, dict) and name in mark:
        return mark[name]
    return None


def _safe_literal(value):
    return repr(value)


def _is_probably_vegalite(spec):
    if not isinstance(spec, dict):
        return False
    if "mark" in spec and ("encoding" in spec or "data" in spec):
        return True
    if any(key in spec for key in ("layer", "hconcat", "vconcat", "concat", "facet")):
        return True
    return "$schema" in spec and ("mark" in spec or "encoding" in spec)


def parse_vegalite_spec(value):
    """Return a Vega-Lite spec dict from a dict/string, or None if not parseable."""
    if isinstance(value, dict):
        if _is_probably_vegalite(value):
            return value
        nested = value.get("spec")
        if isinstance(nested, dict) and _is_probably_vegalite(nested):
            return nested
        return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    candidates = []
    fenced = re.findall(
        r"```(?:json|vega-lite|vegalite)?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )
    candidates.extend(block.strip() for block in fenced if block.strip())
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            spec = parse_vegalite_spec(parsed)
            if spec is not None:
                return spec

        for match in re.finditer(r"[\{\[]", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except Exception:
                try:
                    parsed = ast.literal_eval(candidate[match.start() :])
                except Exception:
                    continue
            spec = parse_vegalite_spec(parsed)
            if spec is not None:
                return spec

    return None

def get_sample_vegalite_spec(sample):
    for key in VEGALITE_SPEC_KEYS:
        if key not in sample:
            continue
        spec = parse_vegalite_spec(sample[key])
        if spec is not None and is_supported_vegalite_spec(spec):
            return canonicalize_vegalite_spec(spec)

    code = sample.get("code")
    if code:
        spec = matplotlib_code_to_vegalite(code)
        return spec
    return None


def sample_target_text(sample):
    spec = get_sample_vegalite_spec(sample)
    if spec is None:
        return None
    return canonical_json_dumps(spec)


def canonicalize_vegalite_spec(spec):
    """Normalize Vega-Lite specs for stable training/evaluation."""
    return _canonicalize(parse_vegalite_spec(spec) or spec)


def canonical_json_dumps(spec):
    return json.dumps(canonicalize_vegalite_spec(spec), indent=2, sort_keys=True)


def _canonicalize(value):
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def infer_supported_plot_type(spec):
    """Classify a spec into the project-supported plot subset."""
    spec = parse_vegalite_spec(spec)
    if spec is None:
        return None

    # Handle composite specs (hconcat/vconcat) used for subplots
    for key in ("hconcat", "vconcat", "concat"):
        if key in spec:
            children = spec[key]
            if isinstance(children, list) and children:
                types = [infer_supported_plot_type(child) for child in children]
                if all(t is not None and t in SUPPORTED_PLOT_SUBSET for t in types):
                    return types[0]  # Return the first child's type for classification
            return None

    if "layer" in spec:
        marks = [_mark_type(layer) for layer in spec.get("layer", [])]
        if marks and all(mark == "line" for mark in marks):
            return "multi-line"
        if marks and all(mark == "bar" for mark in marks):
            return "grouped-bar"
        # Mixed layers containing supported marks
        if marks and all(m in {"point", "circle", "line", "bar", "area", "rule"} for m in marks if m):
            return "scatter"  # generic fallback for mixed layers
        return None

    mark = _mark_type(spec)
    enc = spec.get("encoding", {})
    if mark in {"point", "circle"}:
        return "scatter"
    if mark == "line":
        # Check for step interpolation
        mark_obj = spec.get("mark")
        if isinstance(mark_obj, dict) and mark_obj.get("interpolate", "").startswith("step"):
            return "step"
        return "multi-line" if _field(spec, "color") else "line"
    if mark == "area":
        return "area"
    if mark == "bar":
        if _channel(spec, "x").get("bin") or _channel(spec, "y").get("bin"):
            return "histogram"
        if _field(spec, "color") or "xOffset" in enc or "yOffset" in enc:
            return "grouped-bar"
        return "bar"
    if mark == "rect":
        return "heatmap"
    if mark == "boxplot":
        return "boxplot"
    if mark == "arc":
        return "pie"
    return None


def is_supported_vegalite_spec(spec):
    parsed = parse_vegalite_spec(spec)
    if parsed is None:
        return False
    # Handle composite specs
    for key in ("hconcat", "vconcat", "concat"):
        if key in parsed:
            children = parsed[key]
            if isinstance(children, list) and children:
                return all(
                    infer_supported_plot_type(child) in SUPPORTED_PLOT_SUBSET
                    for child in children
                )
            return False
    return infer_supported_plot_type(parsed) in SUPPORTED_PLOT_SUBSET


def save_vegalite_spec(spec, image_path=None, output_dir=VEGA_SPEC_DIR):
    """Save a canonical Vega-Lite spec under train_data/vega_specs."""
    spec = canonicalize_vegalite_spec(spec)
    if not is_supported_vegalite_spec(spec):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_path:
        stem = Path(str(image_path)).stem
    else:
        stem = hashlib.sha1(canonical_json_dumps(spec).encode("utf-8")).hexdigest()[:12]
    out_path = output_dir / f"{stem}.json"
    out_path.write_text(canonical_json_dumps(spec) + "\n", encoding="utf-8")
    return str(out_path)


def ensure_sample_vegalite_spec(sample, save=True, output_dir=VEGA_SPEC_DIR):
    """Attach/save a programmatically generated or existing supported Vega-Lite spec."""
    spec = get_sample_vegalite_spec(sample)
    if spec is None:
        return None
    sample["vega_spec"] = spec
    sample["vega_lite_spec"] = spec
    if save:
        saved_path = save_vegalite_spec(spec, sample.get("image"), output_dir=output_dir)
        if saved_path:
            sample["vega_lite_spec_path"] = saved_path
    return spec


def generate_and_save_vegalite_specs(samples, output_dir=VEGA_SPEC_DIR):
    """Programmatically generate/save Vega-Lite specs for supported samples."""
    converted = []
    skipped = []
    for index, sample in enumerate(samples):
        spec = ensure_sample_vegalite_spec(sample, save=True, output_dir=output_dir)
        if spec is None:
            skipped.append(index)
        else:
            converted.append(sample)
    return converted, skipped


def evaluate_vegalite_ir(reference_spec, generated_spec):
    """Evaluate intermediate Vega-Lite representation accuracy."""
    ref = canonicalize_vegalite_spec(reference_spec)
    gen = canonicalize_vegalite_spec(generated_spec)
    ref_fields = _flatten_json(ref)
    gen_fields = _flatten_json(gen)
    ref_items = set(ref_fields.items())
    gen_items = set(gen_fields.items())
    matched = ref_items & gen_items

    precision = len(matched) / len(gen_items) if gen_items else 0.0
    recall = len(matched) / len(ref_items) if ref_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    ref_paths = set(ref_fields)
    gen_paths = set(gen_fields)
    path_overlap = len(ref_paths & gen_paths) / len(ref_paths | gen_paths) if ref_paths or gen_paths else 1.0
    value_overlap = len(matched) / len(ref_items | gen_items) if ref_items or gen_items else 1.0

    return {
        "exact_match": ref == gen,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "tree_similarity": 0.5 * path_overlap + 0.5 * value_overlap,
    }


def _flatten_json(value, prefix=""):
    if isinstance(value, dict):
        fields = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            fields.update(_flatten_json(item, child_prefix))
        return fields
    if isinstance(value, list):
        fields = {}
        for idx, item in enumerate(value):
            fields.update(_flatten_json(item, f"{prefix}[{idx}]"))
        return fields
    return {prefix: json.dumps(value, sort_keys=True)}


def matplotlib_code_to_vegalite(code):
    """Generate a supported Vega-Lite spec from simple Matplotlib code without an LLM."""
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    env = {}
    plot_calls = []
    title = None
    x_label = None
    y_label = None
    subplot_info = None  # Track subplot configuration

    # Ordered walk through the top-level statements (preserves assignment order)
    stmts = _collect_stmts_ordered(tree)
    for node in stmts:
        _process_node(node, env, plot_calls)

    # Extract title/labels from all call nodes in the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"title", "set_title"} and node.args:
                title = _literal_or_name(node.args[0], env) or title
            elif name in {"xlabel", "set_xlabel"} and node.args:
                x_label = _literal_or_name(node.args[0], env) or x_label
            elif name in {"ylabel", "set_ylabel"} and node.args:
                y_label = _literal_or_name(node.args[0], env) or y_label
            elif name == "set":
                title = _kw(node, "title", env, title)
                x_label = _kw(node, "xlabel", env, x_label)
                y_label = _kw(node, "ylabel", env, y_label)
            elif name == "subplots":
                subplot_info = _parse_subplot_call(node, env)

    if not plot_calls:
        return None

    same_kind = {name for name, _ in plot_calls}

    # Try single-axis conversion first
    spec = _try_single_axis_conversion(plot_calls, same_kind, env)

    # If single-axis failed, try subplot conversion
    if spec is None and subplot_info is not None:
        spec = _try_subplot_conversion(tree, env, subplot_info)

    if spec is None:
        return None

    # Apply title/labels to top-level or first sub-spec
    _apply_labels(spec, title, x_label, y_label)

    spec = canonicalize_vegalite_spec(spec)
    return spec if is_supported_vegalite_spec(spec) else None


def _collect_stmts_ordered(tree):
    """Collect all statements in order, recursively entering function defs, for loops, with blocks."""
    stmts = []
    for node in ast.iter_child_nodes(tree):
        stmts.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                stmts.append(child)
        elif isinstance(node, ast.For):
            for child in node.body:
                stmts.append(child)
        elif isinstance(node, ast.With):
            for child in node.body:
                stmts.append(child)
        elif isinstance(node, ast.If):
            for child in node.body:
                stmts.append(child)
            for child in node.orelse:
                stmts.append(child)
        elif isinstance(node, ast.Try):
            for child in node.body:
                stmts.append(child)
    return stmts


def _process_node(node, env, plot_calls):
    """Process a single AST node for env assignment and plot call collection."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            value = _literal_or_name(node.value, env)
            if value is not None:
                env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            value = _literal_or_name(node.value, env)
            if isinstance(value, (list, tuple)) and len(value) == len(target.elts):
                for t, item in zip(target.elts, value):
                    if isinstance(t, ast.Name):
                        env[t.id] = item
        elif isinstance(target, ast.Subscript):
            # Handle dict subscript assignment: data['key'] = value
            _handle_subscript_assign(target, node.value, env)
    elif isinstance(node, ast.AugAssign):
        # Handle += patterns (e.g., bottom += weight_count)
        if isinstance(node.target, ast.Name):
            current = env.get(node.target.id)
            addition = _literal_or_name(node.value, env)
            if current is not None and addition is not None:
                try:
                    if isinstance(node.op, ast.Add):
                        env[node.target.id] = _to_builtin(np.asarray(current) + np.asarray(addition))
                except Exception:
                    pass
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        _collect_plot_call(node.value, env, plot_calls)
    elif isinstance(node, ast.For):
        _process_for_loop(node, env, plot_calls)
    elif isinstance(node, ast.With):
        for child in node.body:
            _process_node(child, env, plot_calls)


def _handle_subscript_assign(target, value_node, env):
    """Handle assignments like data['key'] = expr."""
    if isinstance(target.value, ast.Name):
        dict_name = target.value.id
        if dict_name in env and isinstance(env[dict_name], dict):
            key = _literal_or_name(target.slice, env)
            if key is not None:
                val = _literal_or_name(value_node, env)
                if val is not None:
                    env[dict_name][key] = val


def _collect_plot_call(call_node, env, plot_calls):
    """Collect recognized plot calls."""
    name = _call_name(call_node)
    PLOT_METHODS = {
        "plot", "scatter", "bar", "barh", "hist", "imshow", "pcolormesh",
        "boxplot", "fill_between", "heatmap", "stem", "errorbar", "pie",
        "step", "stairs", "stackplot", "matshow",
        "semilogx", "semilogy", "loglog",
    }
    if name in PLOT_METHODS:
        plot_calls.append((name, call_node))


def _process_for_loop(for_node, env, plot_calls):
    """Process for-loop bodies to detect multi-series plot patterns."""
    # Try to collect plot calls from the loop body
    for child in for_node.body:
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
            _collect_plot_call(child.value, env, plot_calls)
        elif isinstance(child, ast.Assign):
            _process_node(child, env, plot_calls)


def _parse_subplot_call(node, env):
    """Extract subplot configuration from plt.subplots(...) call."""
    args = [_literal_or_name(arg, env) for arg in node.args]
    nrows = args[0] if len(args) > 0 and args[0] is not None else 1
    ncols = args[1] if len(args) > 1 and args[1] is not None else 1
    # Also check keyword args
    for kw in node.keywords:
        if kw.arg == "nrows":
            val = _literal_or_name(kw.value, env)
            if val is not None:
                nrows = val
        elif kw.arg == "ncols":
            val = _literal_or_name(kw.value, env)
            if val is not None:
                ncols = val
    try:
        return {"nrows": int(nrows), "ncols": int(ncols)}
    except (TypeError, ValueError):
        return None


def _try_single_axis_conversion(plot_calls, same_kind, env):
    """Try converting plot calls to a single-axis Vega-Lite spec."""
    if len(plot_calls) > 1 and same_kind <= {"plot"}:
        return _multi_line_calls_to_spec(plot_calls, env)
    if len(plot_calls) > 1 and same_kind <= {"bar"}:
        return _grouped_bar_calls_to_spec(plot_calls, env)
    if len(plot_calls) > 1 and same_kind <= {"plot", "fill_between"}:
        # Extract just the plot calls for multi-line, ignore fill_between
        line_calls = [(n, nd) for n, nd in plot_calls if n == "plot"]
        if len(line_calls) >= 1:
            return _multi_line_calls_to_spec(line_calls, env) if len(line_calls) > 1 else _single_plot_call_to_spec("plot", line_calls[0][1], env)
    if len(plot_calls) > 1 and same_kind <= {"scatter"}:
        # Multiple scatter calls -> combined scatter
        return _multi_scatter_calls_to_spec(plot_calls, env)
    if len(plot_calls) == 1:
        name, node = plot_calls[0]
        return _single_plot_call_to_spec(name, node, env)
    # Mixed calls with one main type - try fallback to first supported call
    if len(plot_calls) > 1:
        for name, node in plot_calls:
            spec = _single_plot_call_to_spec(name, node, env)
            if spec is not None:
                return spec
    return None


def _try_subplot_conversion(tree, env, subplot_info):
    """Try to convert simple subplot layouts to hconcat/vconcat specs."""
    nrows = subplot_info["nrows"]
    ncols = subplot_info["ncols"]

    # Only handle simple 1D layouts (1×N or N×1) with small N
    if nrows > 1 and ncols > 1:
        return None
    n_panels = max(nrows, ncols)
    if n_panels > 6 or n_panels < 2:
        return None

    # Find all plot calls in the code, grouped by axis variable
    axis_plots = _extract_subplot_plot_calls(tree, env, n_panels)
    if not axis_plots or len(axis_plots) < 2:
        return None

    # Convert each panel
    panels = []
    for ax_key in sorted(axis_plots.keys()):
        calls = axis_plots[ax_key]
        if not calls:
            continue
        same_kind = {name for name, _ in calls}
        panel_spec = _try_single_axis_conversion(calls, same_kind, env)
        if panel_spec is None:
            return None  # If any panel fails, abort
        panels.append(panel_spec)

    if len(panels) < 2:
        return None

    container_key = "hconcat" if ncols > 1 else "vconcat"
    return {container_key: panels}


def _extract_subplot_plot_calls(tree, env, n_panels):
    """Extract plot calls grouped by axis index for subplot conversion."""
    axis_plots = {}

    # Pattern: axs[0].plot(...), axs[1].scatter(...), etc.
    # Or: ax1.plot(...), ax2.plot(...)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in {
            "plot", "scatter", "bar", "barh", "hist", "imshow", "pcolormesh",
            "boxplot", "fill_between", "stem", "errorbar", "pie", "step",
        }:
            continue

        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        ax_key = _resolve_axis_key(func.value)
        if ax_key is not None:
            axis_plots.setdefault(ax_key, []).append((name, node))

    return axis_plots if axis_plots else None


def _resolve_axis_key(node):
    """Resolve axis reference to a sortable key."""
    # axs[0], axes[1], etc.
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {"axs", "axes", "ax"}:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                return node.slice.value
    # ax1, ax2, etc.
    if isinstance(node, ast.Name):
        name = node.id
        if re.match(r'^ax\d+$', name):
            return int(name[2:])
    return None


def _apply_labels(spec, title, x_label, y_label):
    """Apply title and axis labels to a spec."""
    if "hconcat" in spec or "vconcat" in spec or "concat" in spec:
        if title:
            spec["title"] = str(title)
        return

    if title:
        spec["title"] = str(title)
    enc = spec.get("encoding", {})
    if x_label and "x" in enc:
        enc["x"]["title"] = str(x_label)
    if y_label and "y" in enc:
        enc["y"]["title"] = str(y_label)


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _call_full_name(node):
    parts = []
    func = node.func if isinstance(node, ast.Call) else node
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts)) if parts else None


def _literal_or_name(node, env):
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal_or_name(item, env) for item in node.elts]
        if any(item is None for item in items):
            return None
        return items
    if isinstance(node, ast.Dict):
        keys = [_literal_or_name(k, env) for k in node.keys]
        values = [_literal_or_name(v, env) for v in node.values]
        if any(k is None for k in keys):
            return None
        return dict(zip(keys, values))
    if isinstance(node, ast.Subscript):
        return _eval_subscript(node, env)
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, env)
    if isinstance(node, ast.UnaryOp):
        value = _literal_or_name(node.operand, env)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return _apply_unary(value, lambda item: -item)
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Attribute):
        # Handle np.pi, np.e, np.inf etc.
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy", "math"}:
            attr = node.attr
            if attr == "pi":
                return np.pi
            if attr == "e":
                return np.e
            if attr == "inf":
                return np.inf
            if attr == "nan":
                return np.nan
        # Handle obj.attr access on env objects
        receiver = _literal_or_name(node.value, env)
        if receiver is not None and isinstance(receiver, dict):
            return receiver.get(node.attr)
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name in {"array", "asarray", "list", "tuple"} and node.args:
            return _to_builtin(_literal_or_name(node.args[0], env))
        if name in {"arange", "linspace"}:
            return _eval_numeric_constructor(name, node, env)
        value = _eval_supported_call(node, env)
        if value is not None:
            return _to_builtin(value)
    if isinstance(node, ast.Set):
        return [_literal_or_name(item, env) for item in node.elts]
    if isinstance(node, ast.IfExp):
        # Handle ternary expressions: x if cond else y
        # We can't evaluate the condition, so try body first
        return _literal_or_name(node.body, env)
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _to_builtin(value):
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _to_numpy(value):
    if np is None:
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _apply_unary(value, op):
    if isinstance(value, list):
        return [_apply_unary(item, op) for item in value]
    if isinstance(value, tuple):
        return [_apply_unary(item, op) for item in value]
    try:
        return op(value)
    except Exception:
        return None


def _eval_binop(node, env):
    left = _literal_or_name(node.left, env)
    right = _literal_or_name(node.right, env)
    if left is None or right is None:
        return None
    if np is not None:
        left_value = _to_numpy(left)
        right_value = _to_numpy(right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a ** b,
            ast.Mod: lambda a, b: a % b,
        }
        for op_type, op in ops.items():
            if isinstance(node.op, op_type):
                try:
                    return _to_builtin(op(left_value, right_value))
                except Exception:
                    return None
    try:
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    except Exception:
        return None
    return None


def _eval_subscript(node, env):
    value = _literal_or_name(node.value, env)
    if value is None:
        return None
    try:
        if isinstance(node.slice, ast.Slice):
            lower = _literal_or_name(node.slice.lower, env) if node.slice.lower else None
            upper = _literal_or_name(node.slice.upper, env) if node.slice.upper else None
            step = _literal_or_name(node.slice.step, env) if node.slice.step else None
            return value[slice(lower, upper, step)]
        index = _literal_or_name(node.slice, env)
        return value[index]
    except Exception:
        return None


def _eval_supported_call(node, env):
    full_name = _call_full_name(node)
    name = _call_name(node)
    args = [_literal_or_name(arg, env) for arg in node.args]
    kwargs = {keyword.arg: _literal_or_name(keyword.value, env) for keyword in node.keywords if keyword.arg}

    # Helper: check that no required args are None
    def _args_ok(arg_list=args, kw_dict=kwargs):
        return all(a is not None for a in arg_list) and all(v is not None for v in kw_dict.values())

    if name == "len" and args and args[0] is not None:
        try:
            return len(args[0])
        except Exception:
            return None

    if name == "int" and args and args[0] is not None:
        try:
            return int(args[0])
        except Exception:
            return None

    if name == "float" and args and args[0] is not None:
        try:
            return float(args[0])
        except Exception:
            return None

    if name == "str" and args and args[0] is not None:
        try:
            return str(args[0])
        except Exception:
            return None

    if name == "range":
        safe_args = [a for a in args if a is not None]
        if safe_args:
            try:
                return list(range(*[int(a) for a in safe_args]))
            except Exception:
                return None

    if name == "dict" and not args:
        return dict(**{k: v for k, v in kwargs.items() if v is not None})

    # Handle dict.keys(), dict.values(), dict.items()
    if isinstance(node.func, ast.Attribute) and name in {"keys", "values", "items"}:
        receiver = _literal_or_name(node.func.value, env)
        if isinstance(receiver, dict):
            try:
                result = getattr(receiver, name)()
                return list(result)
            except Exception:
                return None

    # Handle list/tuple constructors from generators or iterables
    if name in {"list", "tuple"} and args and args[0] is not None:
        try:
            return list(args[0]) if name == "list" else list(args[0])
        except Exception:
            return None

    if np is None:
        return None

    # Guard all numpy constructors against None args
    if full_name in {"np.arange", "numpy.arange"} or name == "arange":
        if not _args_ok():
            return None
        try:
            return np.arange(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.linspace", "numpy.linspace"} or name == "linspace":
        if not _args_ok():
            return None
        try:
            return np.linspace(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.zeros", "numpy.zeros"} or name == "zeros":
        if not _args_ok():
            return None
        try:
            return np.zeros(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.ones", "numpy.ones"} or name == "ones":
        if not _args_ok():
            return None
        try:
            return np.ones(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.full", "numpy.full"} or name == "full":
        if not _args_ok():
            return None
        try:
            return np.full(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.repeat", "numpy.repeat"} or name == "repeat":
        if not _args_ok():
            return None
        try:
            return np.repeat(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.meshgrid", "numpy.meshgrid"} or name == "meshgrid":
        if not _args_ok():
            return None
        try:
            return np.meshgrid(*args, **kwargs)
        except Exception:
            return None
    if full_name in {"np.array", "numpy.array"} or (name == "array" and args):
        if not _args_ok():
            return None
        try:
            return np.array(*args, **kwargs)
        except Exception:
            return None

    # Numpy aggregation functions
    if full_name in {"np.sum", "numpy.sum"} or (name == "sum" and full_name and "np" in full_name):
        if args and args[0] is not None:
            try:
                return np.sum(args[0])
            except Exception:
                return None
    if full_name in {"np.mean", "numpy.mean"} or (name == "mean" and full_name and "np" in full_name):
        if args and args[0] is not None:
            try:
                return np.mean(args[0])
            except Exception:
                return None
    if full_name in {"np.cumsum", "numpy.cumsum"} or name == "cumsum":
        if args and args[0] is not None:
            try:
                return np.cumsum(args[0])
            except Exception:
                return None
    if full_name in {"np.diff", "numpy.diff"} or name == "diff":
        if args and args[0] is not None:
            try:
                return np.diff(*[a for a in args if a is not None])
            except Exception:
                return None
    if full_name in {"np.concatenate", "numpy.concatenate"} or name == "concatenate":
        if args and args[0] is not None:
            try:
                return np.concatenate(*args, **kwargs)
            except Exception:
                return None

    # Ufuncs - GUARDED against None
    ufuncs = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "log": np.log,
        "log10": np.log10,
        "abs": np.abs,
        "arctan2": np.arctan2,
        "arctan": np.arctan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
    }
    if name in ufuncs and args:
        if any(a is None for a in args):
            return None
        try:
            return ufuncs[name](*args, **kwargs)
        except Exception:
            return None

    # Random functions - GUARDED against None
    rng = np.random.default_rng(0)
    if full_name in {"np.random.seed", "numpy.random.seed"}:
        return None  # Ignore seed calls, return None is fine
    if full_name in {"np.random.rand", "numpy.random.rand"}:
        if any(a is None for a in args):
            return None
        try:
            return rng.random(tuple(int(arg) for arg in args)) if len(args) > 1 else rng.random(int(args[0]) if args else None)
        except Exception:
            return None
    if full_name in {"np.random.randn", "numpy.random.randn"}:
        if any(a is None for a in args):
            return None
        try:
            size = tuple(int(arg) for arg in args) if len(args) > 1 else (int(args[0]) if args else None)
            return rng.standard_normal(size)
        except Exception:
            return None
    if full_name in {"np.random.randint", "numpy.random.randint"}:
        if any(a is None for a in args):
            return None
        try:
            low = int(args[0]) if len(args) > 0 else 0
            high = int(args[1]) if len(args) > 1 else None
            size = args[2] if len(args) > 2 else kwargs.get("size")
            if high is None:
                high = low
                low = 0
            if size is not None:
                size = int(size) if not isinstance(size, (list, tuple)) else tuple(int(s) for s in size)
            return rng.integers(low=low, high=high, size=size)
        except Exception:
            return None
    if full_name in {"np.random.normal", "numpy.random.normal"}:
        try:
            loc = args[0] if len(args) > 0 and args[0] is not None else kwargs.pop("loc", 0.0)
            scale = args[1] if len(args) > 1 and args[1] is not None else kwargs.pop("scale", 1.0)
            size = args[2] if len(args) > 2 and args[2] is not None else kwargs.pop("size", None)
            if loc is None:
                loc = 0.0
            if scale is None:
                scale = 1.0
            return rng.normal(loc=loc, scale=scale, size=size)
        except Exception:
            return None
    if full_name in {"np.random.uniform", "numpy.random.uniform"}:
        try:
            low = args[0] if len(args) > 0 and args[0] is not None else kwargs.pop("low", 0.0)
            high = args[1] if len(args) > 1 and args[1] is not None else kwargs.pop("high", 1.0)
            size = args[2] if len(args) > 2 and args[2] is not None else kwargs.pop("size", None)
            if low is None:
                low = 0.0
            if high is None:
                high = 1.0
            return rng.uniform(low=low, high=high, size=size)
        except Exception:
            return None
    if full_name in {"np.random.gamma", "numpy.random.gamma"}:
        try:
            shape = args[0] if len(args) > 0 and args[0] is not None else kwargs.pop("shape", 1.0)
            scale = args[1] if len(args) > 1 and args[1] is not None else kwargs.pop("scale", 1.0)
            size = args[2] if len(args) > 2 and args[2] is not None else kwargs.pop("size", None)
            if shape is None:
                shape = 1.0
            if scale is None:
                scale = 1.0
            return rng.gamma(shape=shape, scale=scale, size=size)
        except Exception:
            return None

    # np.pi constant
    if full_name in {"np.pi", "numpy.pi"}:
        return np.pi

    # np.abs / np.abs on args
    if full_name in {"np.abs", "numpy.abs"}:
        if args and args[0] is not None:
            try:
                return np.abs(args[0])
            except Exception:
                return None

    if isinstance(node.func, ast.Attribute):
        receiver = _literal_or_name(node.func.value, env)
        if receiver is not None:
            # Handle dict methods
            if isinstance(receiver, dict):
                if name == "keys":
                    return list(receiver.keys())
                if name == "values":
                    return list(receiver.values())
                if name == "items":
                    return list(receiver.items())
                if name == "get" and args:
                    return receiver.get(args[0], args[1] if len(args) > 1 else None)

            # Handle list methods
            if isinstance(receiver, list):
                if name == "append" and args:
                    return None  # side-effect, can't return
                if name == "copy":
                    return list(receiver)

            try:
                array = np.asarray(receiver)
                if name == "flatten":
                    return array.flatten()
                if name == "ravel":
                    return array.ravel()
                if name == "reshape":
                    if not _args_ok():
                        return None
                    return array.reshape(*args)
                if name == "mean":
                    return array.mean(**{k: v for k, v in kwargs.items() if v is not None})
                if name == "min":
                    return array.min(**{k: v for k, v in kwargs.items() if v is not None})
                if name == "max":
                    return array.max(**{k: v for k, v in kwargs.items() if v is not None})
                if name == "sum":
                    return array.sum(**{k: v for k, v in kwargs.items() if v is not None})
                if name == "cumsum":
                    return array.cumsum()
                if name == "tolist":
                    return array.tolist()
                if name == "T":
                    return array.T
            except Exception:
                return None

    return None


def _eval_numeric_constructor(name, node, env):
    args = [_literal_or_name(arg, env) for arg in node.args]
    if any(arg is None for arg in args):
        return None
    try:
        if name == "arange":
            if len(args) == 1:
                start, stop, step = 0, args[0], 1
            elif len(args) == 2:
                start, stop, step = args[0], args[1], 1
            elif len(args) >= 3:
                start, stop, step = args[0], args[1], args[2]
            else:
                return None
            values = []
            current = start
            while (step > 0 and current < stop) or (step < 0 and current > stop):
                values.append(current)
                current += step
            return values
        if name == "linspace" and len(args) >= 2:
            start, stop = float(args[0]), float(args[1])
            count = int(args[2]) if len(args) >= 3 else 50
            if count <= 1:
                return [start]
            step = (stop - start) / (count - 1)
            return [start + i * step for i in range(count)]
    except Exception:
        return None
    return None


def _kw(node, name, env, default=None):
    for keyword in node.keywords:
        if keyword.arg == name:
            value = _literal_or_name(keyword.value, env)
            return default if value is None else value
    return default


def _as_list(value):
    if value is None:
        return None
    value = _to_builtin(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return None


def _paired_records(x_values, y_values, x_name="x", y_name="y"):
    x_values = _as_list(x_values)
    y_values = _as_list(y_values)
    if y_values is None:
        return None
    if x_values is None:
        x_values = list(range(len(y_values)))
    if len(x_values) != len(y_values):
        return None
    return [{x_name: x, y_name: y} for x, y in zip(x_values, y_values)]


def _single_plot_call_to_spec(name, node, env):
    if name == "plot":
        if not node.args:
            return None
        # Handle plt.plot(range(10)) and plt.plot(data)
        if len(node.args) == 1:
            arg_val = _literal_or_name(node.args[0], env)
            # Handle range() objects
            if isinstance(arg_val, range):
                arg_val = list(arg_val)
            y_values = arg_val
            x_values = None
        else:
            x_values = _literal_or_name(node.args[0], env)
            y_values = _literal_or_name(node.args[1], env)
            if isinstance(x_values, range):
                x_values = list(x_values)
            if isinstance(y_values, range):
                y_values = list(y_values)
        records = _paired_records(x_values, y_values)
        if records is None:
            return None
        return _xy_spec(records, "line")

    if name == "scatter":
        x_arg = node.args[0] if len(node.args) > 0 else None
        y_arg = node.args[1] if len(node.args) > 1 else None
        x_values = _literal_or_name(x_arg, env) if x_arg is not None else _kw(node, "x", env)
        y_values = _literal_or_name(y_arg, env) if y_arg is not None else _kw(node, "y", env)

        # Handle scatter('col1', 'col2', data=data_dict) pattern
        data_dict = _kw(node, "data", env)
        if isinstance(x_values, str) and isinstance(y_values, str) and isinstance(data_dict, dict):
            x_key, y_key = x_values, y_values
            x_data = _as_list(data_dict.get(x_key))
            y_data = _as_list(data_dict.get(y_key))
            if x_data is not None and y_data is not None:
                records = _paired_records(x_data, y_data)
                if records is not None:
                    return _xy_spec(records, "point")

        if x_values is None or y_values is None:
            return None
        records = _paired_records(x_values, y_values)
        if records is None:
            return None
        return _xy_spec(records, "point")

    if name in {"bar", "barh"}:
        x_arg = node.args[0] if len(node.args) > 0 else None
        y_arg = node.args[1] if len(node.args) > 1 else None
        x_values = _literal_or_name(x_arg, env) if x_arg is not None else _kw(node, "x", env)
        y_values = _literal_or_name(y_arg, env) if y_arg is not None else _kw(node, "height", env)

        # Handle dict.keys() / dict.values() patterns for barh
        # e.g., ax.barh(population.keys(), population.values())
        if x_values is None and x_arg is not None and isinstance(x_arg, ast.Call):
            call_name = _call_name(x_arg)
            if call_name in {"keys", "values"} and isinstance(x_arg.func, ast.Attribute):
                dict_val = _literal_or_name(x_arg.func.value, env)
                if isinstance(dict_val, dict):
                    x_values = list(dict_val.keys()) if call_name == "keys" else list(dict_val.values())
        if y_values is None and y_arg is not None and isinstance(y_arg, ast.Call):
            call_name = _call_name(y_arg)
            if call_name in {"keys", "values"} and isinstance(y_arg.func, ast.Attribute):
                dict_val = _literal_or_name(y_arg.func.value, env)
                if isinstance(dict_val, dict):
                    y_values = list(dict_val.keys()) if call_name == "keys" else list(dict_val.values())

        if x_values is None or y_values is None:
            return None
        records = _paired_records(x_values, y_values, "category", "value")
        if records is None:
            return None
        spec = _xy_spec(records, "bar", x_field="category", y_field="value", x_type="nominal")
        if name == "barh":
            spec["encoding"]["x"], spec["encoding"]["y"] = spec["encoding"]["y"], spec["encoding"]["x"]
        return spec

    if name == "hist":
        if not node.args:
            return None
        values = _as_list(_literal_or_name(node.args[0], env))
        if values is None:
            return None
        bins = _kw(node, "bins", env)
        enc_x = {"bin": True, "field": "value", "type": "quantitative"}
        if isinstance(bins, int):
            enc_x["bin"] = {"maxbins": bins}
        return {
            "data": {"values": [{"value": value} for value in values]},
            "mark": "bar",
            "encoding": {
                "x": enc_x,
                "y": {"aggregate": "count", "type": "quantitative"},
            },
        }

    if name in {"imshow", "heatmap", "pcolormesh"}:
        if not node.args:
            return None
        matrix_arg = node.args[-1] if name == "pcolormesh" and len(node.args) >= 3 else node.args[0]
        matrix = _literal_or_name(matrix_arg, env)
        return _matrix_to_heatmap_spec(matrix)

    if name == "boxplot":
        if not node.args:
            return None
        return _boxplot_to_spec(_literal_or_name(node.args[0], env))

    if name == "fill_between":
        if len(node.args) < 2:
            return None
        x_values = _literal_or_name(node.args[0], env)
        y1_values = _literal_or_name(node.args[1], env)
        # If there's a third arg (y2), use the midpoint for a single area
        if len(node.args) >= 3:
            y2_values = _literal_or_name(node.args[2], env)
            y1_list = _as_list(y1_values)
            y2_list = _as_list(y2_values)
            if y1_list is not None and y2_list is not None and len(y1_list) == len(y2_list):
                # Create area spec with y and y2 for range
                x_list = _as_list(x_values)
                if x_list is None:
                    x_list = list(range(len(y1_list)))
                if len(x_list) == len(y1_list):
                    records = [{"x": x, "y": y1, "y2": y2} for x, y1, y2 in zip(x_list, y1_list, y2_list)]
                    return {
                        "data": {"values": records},
                        "mark": "area",
                        "encoding": {
                            "x": {"field": "x", "type": "quantitative"},
                            "y": {"field": "y", "type": "quantitative"},
                            "y2": {"field": "y2"},
                        },
                    }
        records = _paired_records(x_values, y1_values)
        if records is None:
            return None
        return _xy_spec(records, "area")

    # --- NEW PLOT TYPES ---

    if name == "stem":
        if not node.args:
            return None
        if len(node.args) >= 2:
            x_values = _literal_or_name(node.args[0], env)
            y_values = _literal_or_name(node.args[1], env)
        else:
            y_values = _literal_or_name(node.args[0], env)
            x_values = None
        records = _paired_records(x_values, y_values)
        if records is None:
            return None
        # Stem plot is best represented as bar chart with narrow width
        return {
            "data": {"values": records},
            "mark": {"type": "bar", "width": 2},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
            },
        }

    if name == "errorbar":
        if not node.args:
            return None
        x_values = _literal_or_name(node.args[0], env) if len(node.args) > 0 else None
        y_values = _literal_or_name(node.args[1], env) if len(node.args) > 1 else None
        if x_values is None or y_values is None:
            return None
        records = _paired_records(x_values, y_values)
        if records is None:
            return None
        # Errorbar is closest to a scatter/point plot in Vega-Lite
        return _xy_spec(records, "point")

    if name == "pie":
        if not node.args:
            return None
        values = _as_list(_literal_or_name(node.args[0], env))
        if values is None:
            return None
        # Get labels if provided
        labels = _kw(node, "labels", env)
        if labels is None:
            labels = [f"slice_{i+1}" for i in range(len(values))]
        labels = _as_list(labels) or [f"slice_{i+1}" for i in range(len(values))]
        if len(labels) != len(values):
            labels = [f"slice_{i+1}" for i in range(len(values))]
        records = [{"category": str(label), "value": val} for label, val in zip(labels, values)]
        return {
            "data": {"values": records},
            "mark": "arc",
            "encoding": {
                "theta": {"field": "value", "type": "quantitative"},
                "color": {"field": "category", "type": "nominal"},
            },
        }

    if name in {"step", "stairs"}:
        if not node.args:
            return None
        if name == "stairs":
            # stairs(values, edges) -> step plot
            values = _as_list(_literal_or_name(node.args[0], env))
            if values is None:
                return None
            if len(node.args) >= 2:
                edges = _as_list(_literal_or_name(node.args[1], env))
                if edges is not None and len(edges) == len(values) + 1:
                    # Use edge midpoints as x
                    x_values = [(edges[i] + edges[i+1]) / 2 for i in range(len(values))]
                else:
                    x_values = list(range(len(values)))
            else:
                x_values = list(range(len(values)))
            records = _paired_records(x_values, values)
        else:
            # step(x, y)
            if len(node.args) >= 2:
                x_values = _literal_or_name(node.args[0], env)
                y_values = _literal_or_name(node.args[1], env)
            else:
                y_values = _literal_or_name(node.args[0], env)
                x_values = None
            records = _paired_records(x_values, y_values)
        if records is None:
            return None
        return {
            "data": {"values": records},
            "mark": {"type": "line", "interpolate": "step-after"},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
            },
        }

    if name == "stackplot":
        if len(node.args) < 2:
            return None
        x_values = _as_list(_literal_or_name(node.args[0], env))
        if x_values is None:
            return None
        records = []
        for idx in range(1, len(node.args)):
            y_values = _as_list(_literal_or_name(node.args[idx], env))
            if y_values is None:
                continue
            label = f"series_{idx}"
            for x, y in zip(x_values, y_values):
                records.append({"x": x, "y": y, "series": label})
        if not records:
            return None
        return {
            "data": {"values": records},
            "mark": "area",
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative", "stack": True},
                "color": {"field": "series", "type": "nominal"},
            },
        }

    return None


def _multi_scatter_calls_to_spec(plot_calls, env):
    """Combine multiple scatter calls into a single spec with color-coded series."""
    records = []
    for idx, (_, node) in enumerate(plot_calls):
        x_arg = node.args[0] if len(node.args) > 0 else None
        y_arg = node.args[1] if len(node.args) > 1 else None
        x_values = _literal_or_name(x_arg, env) if x_arg is not None else _kw(node, "x", env)
        y_values = _literal_or_name(y_arg, env) if y_arg is not None else _kw(node, "y", env)
        label = _kw(node, "label", env, f"series_{idx + 1}")
        series_records = _paired_records(x_values, y_values)
        if series_records is None:
            continue
        for record in series_records:
            record["series"] = str(label)
        records.extend(series_records)
    if not records:
        return None
    return {
        "data": {"values": records},
        "mark": "point",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
            "color": {"field": "series", "type": "nominal"},
        },
    }


def _xy_spec(records, mark, x_field="x", y_field="y", x_type="quantitative", y_type="quantitative"):
    return {
        "data": {"values": records},
        "mark": mark,
        "encoding": {
            "x": {"field": x_field, "type": x_type},
            "y": {"field": y_field, "type": y_type},
        },
    }


def _multi_line_calls_to_spec(plot_calls, env):
    records = []
    for idx, (_, node) in enumerate(plot_calls):
        y_values = _literal_or_name(node.args[0], env) if len(node.args) == 1 else _literal_or_name(node.args[1], env)
        x_values = None if len(node.args) == 1 else _literal_or_name(node.args[0], env)
        label = _kw(node, "label", env, f"series_{idx + 1}")
        series_records = _paired_records(x_values, y_values)
        if series_records is None:
            return None
        for record in series_records:
            record["series"] = str(label)
        records.extend(series_records)
    return {
        "data": {"values": records},
        "mark": "line",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
            "color": {"field": "series", "type": "nominal"},
        },
    }


def _grouped_bar_calls_to_spec(plot_calls, env):
    records = []
    base_categories = None
    for idx, (_, node) in enumerate(plot_calls):
        if len(node.args) < 2:
            return None
        x_values = _as_list(_literal_or_name(node.args[0], env))
        y_values = _as_list(_literal_or_name(node.args[1], env))
        if y_values is None:
            return None
        if base_categories is None:
            base_categories = x_values if x_values and all(isinstance(x, str) for x in x_values) else list(range(len(y_values)))
        label = _kw(node, "label", env, f"series_{idx + 1}")
        if len(base_categories) != len(y_values):
            return None
        for category, value in zip(base_categories, y_values):
            records.append({"category": category, "value": value, "series": str(label)})
    return {
        "data": {"values": records},
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
            "color": {"field": "series", "type": "nominal"},
        },
    }


def _matrix_to_heatmap_spec(matrix):
    matrix = _to_builtin(matrix)
    if not isinstance(matrix, (list, tuple)) or not matrix:
        return None
    records = []
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            return None
        for col_idx, value in enumerate(row):
            records.append({"x": col_idx, "y": row_idx, "value": value})
    return {
        "data": {"values": records},
        "mark": "rect",
        "encoding": {
            "x": {"field": "x", "type": "ordinal"},
            "y": {"field": "y", "type": "ordinal"},
            "color": {"field": "value", "type": "quantitative"},
        },
    }


def _boxplot_to_spec(values):
    values = _as_list(values)
    if values is None:
        return None
    records = []
    if values and all(isinstance(group, (list, tuple)) for group in values):
        for idx, group in enumerate(values):
            for value in group:
                records.append({"group": f"group_{idx + 1}", "value": value})
        x_channel = {"field": "group", "type": "nominal"}
    else:
        records = [{"group": "value", "value": value} for value in values]
        x_channel = {"field": "group", "type": "nominal"}
    return {
        "data": {"values": records},
        "mark": "boxplot",
        "encoding": {
            "x": x_channel,
            "y": {"field": "value", "type": "quantitative"},
        },
    }


class VegaLiteToMatplotlib:
    """Recursive Vega-Lite to executable Matplotlib code translator."""

    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.datasets = self.spec.get("datasets", {}) if isinstance(self.spec, dict) else {}
        self._lines = []
        self._indent = 0

    def emit(self, *lines):
        for line in lines:
            self._lines.append(("    " * self._indent + line) if line else "")

    def push_indent(self):
        self._indent += 1

    def pop_indent(self):
        self._indent = max(0, self._indent - 1)

    def code(self):
        if self._lines:
            return "\n".join(self._lines)

        self.emit("import matplotlib.pyplot as plt")
        self.emit("import numpy as np")
        self.emit("import pandas as pd")
        self.emit("")

        if _has_facet(self.spec):
            self.visit_facet(self.spec)
        elif "layer" in self.spec:
            self.emit("fig, ax = plt.subplots(figsize=(6, 3.6))")
            self.emit("")
            self.visit_layer(self.spec, ax_expr="ax")
            self.emit("fig.tight_layout()")
        else:
            self.emit("fig, ax = plt.subplots(figsize=(6, 3.6))")
            self.emit("")
            self.visit_unit(self.spec, ax_expr="ax")
            self.emit("fig.tight_layout()")

        self.emit("plt.show()")
        return "\n".join(self._lines)

    def visit_unit(self, spec, ax_expr, df_expr=None):
        if df_expr is None:
            df_expr = "df"
            self._emit_data(spec, df_expr)
            self._emit_transforms(spec, df_expr)
        self._dispatch_mark(spec, ax_expr, df_expr)
        self._emit_axes(spec, ax_expr)

    def visit_layer(self, spec, ax_expr):
        outer_data = spec.get("data")
        for i, layer in enumerate(spec["layer"]):
            df_var = f"df_layer_{i}"
            inner = dict(layer)
            inner.setdefault("data", outer_data)
            self._emit_data(inner, df_var)
            self._emit_transforms(inner, df_var)
            self._dispatch_mark(inner, ax_expr, df_var)
        self._emit_axes(spec["layer"][0], ax_expr)
        if spec.get("title"):
            self.emit(f"{ax_expr}.set_title({_safe_literal(_unwrap_title(spec['title']))})")

    def visit_facet(self, spec):
        enc = spec.get("encoding", {})
        col_field = _channel(spec, "column").get("field")
        row_field = _channel(spec, "row").get("field")
        if col_field is None and row_field is None:
            raise NotImplementedError("Facet requires column or row encoding")
        if col_field and row_field:
            raise NotImplementedError("2D faceting with both column and row is not supported")

        self._emit_data(spec, "df")
        self._emit_transforms(spec, "df")

        facet_field = col_field or row_field
        is_column = col_field is not None
        self.emit(f"facet_values = list(dict.fromkeys(df[{facet_field!r}]))")
        if is_column:
            self.emit("fig, axes = plt.subplots(")
            self.emit("    1, len(facet_values),")
            self.emit("    figsize=(3.0 * len(facet_values), 3.2),")
            self.emit("    sharey=True,")
            self.emit(")")
        else:
            self.emit("fig, axes = plt.subplots(")
            self.emit("    len(facet_values), 1,")
            self.emit("    figsize=(5.0, 2.6 * len(facet_values)),")
            self.emit("    sharex=True,")
            self.emit(")")
        self.emit("if len(facet_values) == 1:")
        self.push_indent()
        self.emit("axes = [axes]")
        self.pop_indent()

        self.emit("for ax, facet_val in zip(axes, facet_values):")
        self.push_indent()
        self.emit(f"sub = df[df[{facet_field!r}] == facet_val]")
        inner = dict(spec)
        inner_enc = dict(enc)
        inner_enc.pop("column", None)
        inner_enc.pop("row", None)
        inner["encoding"] = inner_enc
        inner.pop("data", None)
        self.visit_unit(inner, ax_expr="ax", df_expr="sub")
        self.emit(f"ax.set_title(f'{facet_field}={{facet_val}}')")
        self.pop_indent()
        self.emit("fig.tight_layout()")

    def _emit_data(self, spec, df_var):
        data = spec.get("data") or {}
        if "values" in data:
            self.emit(f"{df_var} = pd.DataFrame({json.dumps(data['values'])})")
            return
        if "name" in data and data["name"] in self.datasets:
            self.emit(f"{df_var} = pd.DataFrame({json.dumps(self.datasets[data['name']])})")
            return
        if "url" in data:
            url = data["url"]
            reader = "read_csv" if str(url).lower().endswith(".csv") else "read_json"
            self.emit(f"{df_var} = pd.{reader}({url!r})")
            return
        self.emit(f"{df_var} = pd.DataFrame()  # spec has no inline data")

    def _emit_transforms(self, spec, df_var):
        for transform in spec.get("transform", []):
            if "filter" in transform:
                expr = self._filter_to_pandas_expr(transform["filter"], df_var)
                if expr:
                    self.emit(f"{df_var} = {df_var}[{expr}].reset_index(drop=True)")
                    continue
            self.emit(f"# unsupported transform skipped: {transform!r}")

    def _filter_to_pandas_expr(self, filter_spec, df_var):
        if isinstance(filter_spec, str):
            expr = filter_spec
            expr = re.sub(
                r"datum\.([A-Za-z_][A-Za-z0-9_]*)",
                lambda match: f"{df_var}[{match.group(1)!r}]",
                expr,
            )
            expr = expr.replace("&&", "&").replace("||", "|")
            return expr

        if not isinstance(filter_spec, dict):
            return None

        field = filter_spec.get("field")
        if not field:
            return None
        lhs = f"{df_var}[{field!r}]"
        comparisons = {
            "equal": "==",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
        }
        parts = []
        for key, op in comparisons.items():
            if key in filter_spec:
                parts.append(f"({lhs} {op} {filter_spec[key]!r})")
        if "oneOf" in filter_spec:
            parts.append(f"{lhs}.isin({filter_spec['oneOf']!r})")
        if "range" in filter_spec and len(filter_spec["range"]) == 2:
            lo, hi = filter_spec["range"]
            parts.append(f"({lhs} >= {lo!r}) & ({lhs} <= {hi!r})")
        return " & ".join(parts) if parts else None

    def _dispatch_mark(self, spec, ax_expr, df_expr):
        mark = _mark_type(spec)
        method = getattr(self, f"visit_mark_{mark}", None)
        if method is None:
            raise NotImplementedError(f"mark {mark!r} not supported")
        method(spec, ax_expr, df_expr)

    def visit_mark_bar(self, spec, ax_expr, df_expr):
        enc = spec.get("encoding", {})
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        x_agg = _channel(spec, "x").get("aggregate")
        y_agg = _channel(spec, "y").get("aggregate")
        color_field = _field(spec, "color")

        if _channel(spec, "x").get("bin") and y_agg == "count":
            bins = _channel(spec, "x").get("bin")
            bin_arg = ""
            if isinstance(bins, dict) and isinstance(bins.get("maxbins"), int):
                bin_arg = f", bins={bins['maxbins']}"
            self.emit(f"{ax_expr}.hist({df_expr}[{x_field!r}]{bin_arg})")
            return

        if y_agg in SUPPORTED_AGGREGATES and not x_agg:
            self._emit_aggregated_bar(
                df_expr, ax_expr, x_field, y_field, y_agg, horizontal=False, color_field=color_field
            )
            return
        if x_agg in SUPPORTED_AGGREGATES and not y_agg:
            self._emit_aggregated_bar(
                df_expr, ax_expr, y_field, x_field, x_agg, horizontal=True, color_field=color_field
            )
            return
        if not x_field or not y_field:
            raise NotImplementedError("bar mark requires x/y fields or an aggregate")

        color_value = _value(spec, "color")
        color_arg = f", color={color_value!r}" if color_value is not None else ""
        if color_field is None:
            self.emit(f"{ax_expr}.bar({df_expr}[{x_field!r}], {df_expr}[{y_field!r}]{color_arg})")
            return

        self.emit(f"groups = list(dict.fromkeys({df_expr}[{color_field!r}]))")
        self.emit(f"x_vals = list(dict.fromkeys({df_expr}[{x_field!r}]))")
        self.emit("x_idx = np.arange(len(x_vals))")
        self.emit("width = 0.8 / max(1, len(groups))")
        self.emit("for i, g in enumerate(groups):")
        self.push_indent()
        self.emit(f"sub = {df_expr}[{df_expr}[{color_field!r}] == g]")
        self.emit(f"ys = [sub[sub[{x_field!r}] == xv][{y_field!r}].sum() for xv in x_vals]")
        self.emit(f"{ax_expr}.bar(x_idx + i * width - 0.4 + width / 2, ys, width, label=str(g))")
        self.pop_indent()
        self.emit(f"{ax_expr}.set_xticks(x_idx)")
        self.emit(f"{ax_expr}.set_xticklabels(x_vals)")
        self.emit(f"{ax_expr}.legend(title={color_field!r})")

    def _emit_aggregated_bar(self, df_expr, ax_expr, group_field, value_field, agg, horizontal, color_field=None):
        if not group_field:
            raise NotImplementedError("aggregated bar requires a grouping field")

        if color_field is not None:
            if agg == "count":
                self.emit(
                    f"agg_df = ({df_expr}.groupby([{group_field!r}, {color_field!r}])"
                    ".size().unstack(fill_value=0))"
                )
            else:
                self.emit(
                    f"agg_df = ({df_expr}.groupby([{group_field!r}, {color_field!r}])"
                    f"[{value_field!r}].{agg}().unstack(fill_value=0))"
                )
            self.emit("groups = list(agg_df.columns)")
            self.emit("x_vals = list(agg_df.index)")
            self.emit("x_idx = np.arange(len(x_vals))")
            self.emit("width = 0.8 / max(1, len(groups))")
            self.emit("for i, g in enumerate(groups):")
            self.push_indent()
            offset = "x_idx + i * width - 0.4 + width / 2"
            if horizontal:
                self.emit(f"{ax_expr}.barh({offset}, agg_df[g], width, label=str(g))")
            else:
                self.emit(f"{ax_expr}.bar({offset}, agg_df[g], width, label=str(g))")
            self.pop_indent()
            if horizontal:
                self.emit(f"{ax_expr}.set_yticks(x_idx)")
                self.emit(f"{ax_expr}.set_yticklabels(x_vals)")
            else:
                self.emit(f"{ax_expr}.set_xticks(x_idx)")
                self.emit(f"{ax_expr}.set_xticklabels(x_vals)")
            self.emit(f"{ax_expr}.legend(title={color_field!r})")
            return

        if agg == "count":
            self.emit(f"agg_df = {df_expr}.groupby({group_field!r}).size().reset_index(name='value')")
            value_col = "value"
        else:
            if not value_field:
                raise NotImplementedError(f"{agg} aggregate requires a value field")
            self.emit(f"agg_df = {df_expr}.groupby({group_field!r})[{value_field!r}].{agg}().reset_index()")
            value_col = value_field
        if horizontal:
            self.emit(f"{ax_expr}.barh(agg_df[{group_field!r}], agg_df[{value_col!r}])")
        else:
            self.emit(f"{ax_expr}.bar(agg_df[{group_field!r}], agg_df[{value_col!r}])")

    def visit_mark_point(self, spec, ax_expr, df_expr):
        self._emit_scatter(spec, ax_expr, df_expr)

    def visit_mark_circle(self, spec, ax_expr, df_expr):
        self._emit_scatter(spec, ax_expr, df_expr)

    def _emit_scatter(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        color_field = _field(spec, "color")
        size_field = _field(spec, "size")
        opacity_field = _field(spec, "opacity")
        if not x_field or not y_field:
            raise NotImplementedError("point/circle marks require x and y fields")

        size_arg = ", s=24"
        if size_field:
            size_arg = (
                f", s=({df_expr}[{size_field!r}] / max(1.0, {df_expr}[{size_field!r}].max())) * 200 + 8"
            )
        alpha_arg = ""
        if opacity_field:
            alpha_arg = (
                f", alpha=({df_expr}[{opacity_field!r}] / max(1.0, {df_expr}[{opacity_field!r}].max())).clip(0, 1)"
            )
        color_value = _value(spec, "color")
        color_arg = f", color={color_value!r}" if color_value is not None and color_field is None else ""

        if color_field is None:
            self.emit(f"{ax_expr}.scatter({df_expr}[{x_field!r}], {df_expr}[{y_field!r}]{size_arg}{alpha_arg}{color_arg})")
            return

        self.emit(f"for label, sub in {df_expr}.groupby({color_field!r}):")
        self.push_indent()
        sub_size = ", s=24"
        if size_field:
            sub_size = (
                f", s=(sub[{size_field!r}] / max(1.0, {df_expr}[{size_field!r}].max())) * 200 + 8"
            )
        sub_alpha = ""
        if opacity_field:
            sub_alpha = (
                f", alpha=(sub[{opacity_field!r}] / max(1.0, {df_expr}[{opacity_field!r}].max())).clip(0, 1)"
            )
        self.emit(f"{ax_expr}.scatter(sub[{x_field!r}], sub[{y_field!r}]{sub_size}{sub_alpha}, label=label)")
        self.pop_indent()
        self.emit(f"{ax_expr}.legend(title={color_field!r})")

    def visit_mark_line(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        color_field = _field(spec, "color")
        if not x_field or not y_field:
            raise NotImplementedError("line mark requires x and y fields")

        color_value = _value(spec, "color")
        color_arg = f", color={color_value!r}" if color_value is not None and color_field is None else ""
        if color_field is None:
            self.emit(f"{ax_expr}.plot({df_expr}[{x_field!r}], {df_expr}[{y_field!r}], marker='o'{color_arg})")
            return
        self.emit(f"for label, sub in {df_expr}.groupby({color_field!r}):")
        self.push_indent()
        self.emit(f"{ax_expr}.plot(sub[{x_field!r}], sub[{y_field!r}], marker='o', label=label)")
        self.pop_indent()
        self.emit(f"{ax_expr}.legend(title={color_field!r})")

    def visit_mark_area(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        color_field = _field(spec, "color")
        if not x_field or not y_field:
            raise NotImplementedError("area mark requires x and y fields")

        if color_field is None:
            self.emit(f"{ax_expr}.fill_between({df_expr}[{x_field!r}], {df_expr}[{y_field!r}], alpha=0.4)")
            self.emit(f"{ax_expr}.plot({df_expr}[{x_field!r}], {df_expr}[{y_field!r}])")
            return
        self.emit(f"for label, sub in {df_expr}.groupby({color_field!r}):")
        self.push_indent()
        self.emit(f"{ax_expr}.fill_between(sub[{x_field!r}], sub[{y_field!r}], alpha=0.35, label=label)")
        self.pop_indent()
        self.emit(f"{ax_expr}.legend(title={color_field!r})")

    def visit_mark_tick(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        if not x_field:
            raise NotImplementedError("tick mark requires an x field")
        if y_field:
            self.emit(f"{ax_expr}.scatter({df_expr}[{x_field!r}], {df_expr}[{y_field!r}], marker='|', s=180)")
            return
        self.emit(f"{ax_expr}.scatter({df_expr}[{x_field!r}], np.zeros(len({df_expr})), marker='|', s=180)")
        self.emit(f"{ax_expr}.set_yticks([])")

    def visit_mark_rect(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        color_field = _field(spec, "color")
        if not x_field or not y_field or not color_field:
            raise NotImplementedError("heatmap rect mark requires x, y, and color fields")
        self.emit(
            f"heatmap = {df_expr}.pivot(index={y_field!r}, columns={x_field!r}, values={color_field!r})"
        )
        self.emit(f"im = {ax_expr}.imshow(heatmap.values, aspect='auto')")
        self.emit(f"{ax_expr}.set_xticks(np.arange(len(heatmap.columns)))")
        self.emit(f"{ax_expr}.set_xticklabels(heatmap.columns)")
        self.emit(f"{ax_expr}.set_yticks(np.arange(len(heatmap.index)))")
        self.emit(f"{ax_expr}.set_yticklabels(heatmap.index)")
        self.emit("plt.colorbar(im, ax=ax)")

    def visit_mark_boxplot(self, spec, ax_expr, df_expr):
        x_field = _field(spec, "x")
        y_field = _field(spec, "y")
        if not y_field:
            raise NotImplementedError("boxplot mark requires a y field")
        if x_field:
            self.emit(f"groups = list(dict.fromkeys({df_expr}[{x_field!r}]))")
            self.emit(f"values = [{df_expr}[{df_expr}[{x_field!r}] == g][{y_field!r}] for g in groups]")
            self.emit(f"{ax_expr}.boxplot(values, labels=groups)")
        else:
            self.emit(f"{ax_expr}.boxplot({df_expr}[{y_field!r}])")

    def visit_mark_arc(self, spec, ax_expr, df_expr):
        """Convert Vega-Lite arc mark (pie chart) to matplotlib pie."""
        theta_field = None
        color_field = None
        enc = spec.get("encoding", {})
        if "theta" in enc:
            theta_field = enc["theta"].get("field")
        if "color" in enc:
            color_field = enc["color"].get("field")
        if not theta_field:
            raise NotImplementedError("arc mark requires theta field")
        self.emit(f"{ax_expr}.pie({df_expr}[{theta_field!r}]"
                  f"{', labels=' + df_expr + '[' + repr(color_field) + ']' if color_field else ''}"
                  f", autopct='%1.1f%%')")

    def _emit_axes(self, spec, ax_expr):
        enc = spec.get("encoding", {})
        if "x" in enc:
            self.emit(f"{ax_expr}.set_xlabel({_safe_literal(_enc_label(_channel(spec, 'x')))})")
            scale = _channel(spec, "x").get("scale", {}) or {}
            if scale.get("type") == "log":
                self.emit(f"{ax_expr}.set_xscale('log')")
            if "domain" in scale and len(scale["domain"]) == 2:
                lo, hi = scale["domain"]
                self.emit(f"{ax_expr}.set_xlim({lo!r}, {hi!r})")
        if "y" in enc:
            self.emit(f"{ax_expr}.set_ylabel({_safe_literal(_enc_label(_channel(spec, 'y')))})")
            scale = _channel(spec, "y").get("scale", {}) or {}
            if scale.get("type") == "log":
                self.emit(f"{ax_expr}.set_yscale('log')")
            if "domain" in scale and len(scale["domain"]) == 2:
                lo, hi = scale["domain"]
                self.emit(f"{ax_expr}.set_ylim({lo!r}, {hi!r})")
        if spec.get("title"):
            self.emit(f"{ax_expr}.set_title({_safe_literal(_unwrap_title(spec['title']))})")


def vegalite_to_matplotlib(spec):
    parsed = parse_vegalite_spec(spec) if isinstance(spec, str) else spec
    if parsed is None:
        raise ValueError("Invalid Vega-Lite spec")
    # Handle composite specs (hconcat/vconcat)
    for key in ("hconcat", "vconcat", "concat"):
        if key in parsed:
            children = parsed[key]
            if isinstance(children, list) and len(children) >= 2:
                return _composite_to_matplotlib(parsed, key, children)
    return VegaLiteToMatplotlib(parsed).code()


def _composite_to_matplotlib(spec, layout_key, children):
    """Convert hconcat/vconcat composite specs to subplot-based matplotlib code."""
    n = len(children)
    if layout_key == "hconcat":
        nrows, ncols = 1, n
    else:
        nrows, ncols = n, 1
    lines = [
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "",
        f"fig, axes = plt.subplots({nrows}, {ncols}, figsize=({4*ncols}, {4*nrows}))",
        f"if not hasattr(axes, '__iter__'): axes = [axes]",
    ]
    for i, child in enumerate(children):
        try:
            child_code = VegaLiteToMatplotlib(child).code()
            # Indent the child code under a comment
            lines.append(f"\n# Panel {i+1}")
            lines.append(f"ax = axes[{i}]")
            # Replace 'fig, ax = plt.subplots()' and 'plt.figure()' patterns
            for cline in child_code.split("\n"):
                if "plt.subplots" in cline or "plt.figure" in cline:
                    continue
                if cline.strip().startswith("import "):
                    continue
                lines.append(cline)
        except Exception:
            lines.append(f"# Panel {i+1}: could not convert")
    title = spec.get("title")
    if title:
        lines.append(f"fig.suptitle({json.dumps(title)})")
    lines.append("fig.tight_layout()")
    return "\n".join(lines)


def generated_text_to_matplotlib_code(text):
    spec = parse_vegalite_spec(text)
    if spec is None:
        return None, None
    return vegalite_to_matplotlib(spec), spec
