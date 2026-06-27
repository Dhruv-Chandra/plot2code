import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from common_utils import remove_comments, render_code_to_png
from vegalite_utils import max_workers

# =========================
# CONFIG
# =========================

TIMEOUT = 45
VARIANTS_PER_FILE = 5
MAX_WORKERS = max_workers
BATCH_SIZE = 200
MAX_RETRIES = 2
MAX_TIMEOUT_RETRIES = 0
HEARTBEAT_INTERVAL = 5
SAFE_COLOR_POOL = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
LINEWIDTHS = [1.0, 1.5, 2.0, 2.5, 3.0]
ALPHAS = [0.45, 0.6, 0.75, 0.9]
LINESTYLES = ["-", "--", "-.", ":"]
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis"]
MATPLOTLIB_STYLES = ["default", "bmh", "ggplot", "grayscale", "fast"]
SEABORN_STYLES = ["whitegrid", "darkgrid", "ticks", "white"]
SEABORN_PALETTES = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
SEABORN_FIGURE_LEVEL = (
    "relplot(",
    "catplot(",
    "jointplot(",
    "pairplot(",
    "displot(",
    "clustermap(",
    "FacetGrid(",
)
PLOT_METHOD_NAMES = {
    "plot",
    "scatter",
    "bar",
    "barh",
    "hist",
    "hist2d",
    "imshow",
    "matshow",
    "pcolor",
    "pcolormesh",
    "contour",
    "contourf",
    "tricontour",
    "tricontourf",
    "tripcolor",
    "triplot",
    "quiver",
    "streamplot",
    "barbs",
    "stem",
    "stairs",
    "stackplot",
    "eventplot",
    "errorbar",
    "fill_between",
    "fill_betweenx",
    "pie",
    "boxplot",
    "violinplot",
    "hexbin",
    "ecdf",
    "table",
    "text",
    "annotate",
}
SEABORN_PLOT_METHOD_NAMES = {
    "lineplot",
    "scatterplot",
    "histplot",
    "barplot",
    "boxplot",
    "violinplot",
    "stripplot",
    "swarmplot",
    "pointplot",
    "countplot",
    "kdeplot",
    "ecdfplot",
    "heatmap",
    "regplot",
    "residplot",
    "lmplot",
}
AXES_CONSTRUCTOR_NAMES = {"subplots", "subplot", "add_subplot", "subplot_mosaic", "subfigures"}
BLOCKING_CALL_NAMES = {"ginput", "waitforbuttonpress", "input", "pause", "start_event_loop"}

# Modules that are safe to import during augmented code execution
SAFE_MODULES = {
    "matplotlib", "mpl_toolkits", "numpy", "np", "pandas", "pd",
    "math", "itertools", "functools", "collections", "datetime",
    "io", "os", "sys", "re", "json", "copy", "textwrap",
    "string", "operator", "decimal", "fractions", "statistics",
    "colorsys", "pathlib", "warnings", "csv", "struct",
}

# Plot method names where it's safe to inject marker= / linestyle= kwargs
SAFE_PLOT_METHODS_FOR_KWARGS = {
    "plot", "scatter", "lineplot", "scatterplot",
}


def get_all_py_files(base_path):
    return [p for p in Path(base_path).rglob("*.py") if p.is_file()]


def normalize_code(code: str) -> str:
    cleaned = remove_comments(code)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def code_fingerprint(code: str) -> str:
    return hashlib.sha1(normalize_code(code).encode("utf-8")).hexdigest()


def compute_image_signature(image_path: str) -> str | None:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("L").resize((16, 16), Image.Resampling.LANCZOS))
        median = float(np.median(arr))
        bits = (arr >= median).astype(np.uint8).flatten()
        return "".join(bits.astype(str).tolist())
    except Exception:
        return None


def is_low_information_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("L"))
    except Exception:
        return True

    if arr.size == 0:
        return True

    std = float(arr.std())
    dynamic_range = int(arr.max()) - int(arr.min())
    non_white_ratio = float((arr < 248).mean())
    return std < 10 or dynamic_range < 25 or non_white_ratio < 0.01


def is_different(img1: str, img2: str, threshold: float = 0.985) -> bool:
    try:
        a = np.array(Image.open(img1).convert("L"))
        b = np.array(Image.open(img2).convert("L"))
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        score = ssim(a[:h, :w], b[:h, :w])
        return score < threshold
    except Exception:
        return True


def contains_external_data_dependency(code: str) -> bool:
    external_patterns = [
        "sns.load_dataset(",
        "pd.read_csv(",
        "pd.read_parquet(",
        "pd.read_excel(",
        "pd.read_json(",
        "np.loadtxt(",
        "np.genfromtxt(",
        "requests.",
        "urllib.",
        "open(",
        "Path(",
    ]
    return any(pattern in code for pattern in external_patterns)


def has_unavailable_imports(code: str) -> bool:
    """Check if code imports modules not available in the execution sandbox."""
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module not in SAFE_MODULES:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module not in SAFE_MODULES:
                    return True
    return False


def is_supported_seaborn_code(code: str) -> bool:
    if "sns." not in code and "seaborn" not in code:
        return True
    return not any(pattern in code for pattern in SEABORN_FIGURE_LEVEL)


def extract_called_function_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return names


def count_axes_constructors(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0

    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in AXES_CONSTRUCTOR_NAMES:
            count += 1
        elif isinstance(func, ast.Name) and func.id in AXES_CONSTRUCTOR_NAMES:
            count += 1
    return count


def has_plot_content(code: str) -> bool:
    called = extract_called_function_names(code)
    return bool(called & (PLOT_METHOD_NAMES | SEABORN_PLOT_METHOD_NAMES))


def has_unsupported_complexity(code: str) -> bool:
    complex_patterns = [
        "animation",
        "FuncAnimation",
        "FFMpegWriter",
        "ArtistAnimation",
        "canvas.new_timer",
        "argparse",
        "ArgumentParser",
        "__main__",
        "multiprocessing",
        "mp.Process",
        "Pipe(",
        "plt.ion",
        "xkcd",
        "usetex=True",
        "projection='3d'",
        'projection="3d"',
        "Axes3D",
        "GridSpec",
        "secondary_y",
        "inset_axes",
        # Newer/unsupported matplotlib APIs
        "pie_label(",
        "wedge_labels=",
        "wedge_label_distance=",
        "rotation_mode='xtick'",
        'rotation_mode="xtick"',
        "okabe_ito",
    ]
    return any(pattern in code for pattern in complex_patterns)


def has_blocking_interaction(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return any(f"{name}(" in code for name in BLOCKING_CALL_NAMES)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue

        if name in BLOCKING_CALL_NAMES:
            return True

        if name == "show":
            for keyword in node.keywords:
                if keyword.arg != "block":
                    continue
                try:
                    if ast.literal_eval(keyword.value) is True:
                        return True
                except Exception:
                    continue

        if name == "clabel":
            for keyword in node.keywords:
                if keyword.arg != "manual":
                    continue
                try:
                    if ast.literal_eval(keyword.value) is True:
                        return True
                except Exception:
                    continue

    return False


def should_skip_code(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return True
    if len(stripped) > 18000:
        return True
    if has_unsupported_complexity(code):
        return True
    if has_blocking_interaction(code):
        return True
    if contains_external_data_dependency(code):
        return True
    if has_unavailable_imports(code):
        return True
    if not is_supported_seaborn_code(code):
        return True
    if not has_plot_content(code):
        return True
    if count_axes_constructors(code) > 1:
        return True
    return False


def ensure_imports(code: str) -> str:
    imports = []
    if "np." in code and "import numpy as np" not in code:
        imports.append("import numpy as np")
    if ("plt." in code or "matplotlib" in code) and "import matplotlib.pyplot as plt" not in code:
        imports.append("import matplotlib.pyplot as plt")
    if "sns." in code and "import seaborn as sns" not in code:
        imports.append("import seaborn as sns")
    if "pd." in code and "import pandas as pd" not in code:
        imports.append("import pandas as pd")
    if imports:
        return "\n".join(imports) + "\n" + code
    return code


def sanitize_code_for_execution(code: str) -> str:
    code = ensure_imports(code)
    if "matplotlib.use('Agg')" not in code and 'matplotlib.use("Agg")' not in code:
        code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
    if "plt.show()" not in code:
        code = code.rstrip() + "\nplt.show()\n"
    return code


def is_valid_python(code: str) -> bool:
    try:
        compile(code, "<augment>", "exec")
        return True
    except Exception:
        return False


def extract_colors(code: str):
    patterns = [
        r"color\s*=\s*['\"](.*?)['\"]",
        r"c\s*=\s*['\"](.*?)['\"]",
        r"facecolor\s*=\s*['\"](.*?)['\"]",
        r"edgecolor\s*=\s*['\"](.*?)['\"]",
    ]
    seen = []
    for pattern in patterns:
        for value in re.findall(pattern, code):
            if value and value not in seen and "tab:" not in value:
                seen.append(value)
    return seen


def generate_color_mappings(original_colors, num_variants=4):
    unique_colors = list(dict.fromkeys(original_colors))
    count = len(unique_colors)
    if count == 0 or count > len(SAFE_COLOR_POOL):
        return []

    mappings = []
    used = set()
    attempts = 0
    while len(mappings) < num_variants and attempts < num_variants * 12:
        candidate = tuple(random.sample(SAFE_COLOR_POOL, count))
        attempts += 1
        if candidate in used:
            continue
        used.add(candidate)
        mappings.append(dict(zip(unique_colors, candidate)))
    return mappings


def apply_color_mapping(code: str, color_map: dict) -> str:
    patterns = [
        r"(color\s*=\s*['\"])(.*?)(['\"])",
        r"(c\s*=\s*['\"])(.*?)(['\"])",
        r"(facecolor\s*=\s*['\"])(.*?)(['\"])",
        r"(edgecolor\s*=\s*['\"])(.*?)(['\"])",
    ]

    for pattern in patterns:
        def replace(match):
            current = match.group(2)
            if current in color_map:
                return f"{match.group(1)}{color_map[current]}{match.group(3)}"
            return match.group(0)

        code = re.sub(pattern, replace, code)
    return code


def replace_palette(code: str) -> str:
    if "sns." not in code:
        return code
    palette = random.choice(SEABORN_PALETTES)
    if "palette=" in code:
        return re.sub(r"palette\s*=\s*['\"].*?['\"]", f"palette='{palette}'", code)
    if "sns.set_theme(" in code:
        palette_kwarg = f"palette='{palette}'"
        return re.sub(
            r"sns\.set_theme\((.*?)\)",
            lambda m: f"sns.set_theme({append_kwarg(m.group(1), palette_kwarg)})",
            code,
            count=1,
        )
    return code


def _find_safe_plot_call_insertion_point(code: str, method_names: set[str] | None = None) -> int | None:
    """Find the byte offset of the closing paren of the first safe plot call.

    Uses AST to locate the actual Call node for plot methods, so we never
    accidentally inject kwargs into nested calls like np.sin() or np.random.rand().
    Returns the offset (in the source string) of the closing ')' of the call,
    or None if no suitable call is found.
    """
    if method_names is None:
        method_names = SAFE_PLOT_METHODS_FOR_KWARGS
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name not in method_names:
            continue

        # Find the closing paren of this Call by scanning from end_col_offset
        # ast gives us end_lineno / end_col_offset (1-indexed line, 0-indexed col)
        if node.end_lineno is None or node.end_col_offset is None:
            continue

        # Convert line/col to string offset
        lines = code.splitlines(keepends=True)
        offset = sum(len(lines[i]) for i in range(node.end_lineno - 1)) + node.end_col_offset
        # end_col_offset points one past the closing paren, so the ')' is at offset-1
        if offset > 0 and code[offset - 1] == ')':
            return offset - 1

    return None


def replace_marker(code: str) -> str:
    marker = random.choice(MARKERS)
    if "marker=" in code:
        return re.sub(r"(marker\s*=\s*['\"]).*?(['\"])", rf"\g<1>{marker}\g<2>", code)
    # Use AST to find the actual plot call and inject marker= safely
    insertion = _find_safe_plot_call_insertion_point(code)
    if insertion is not None:
        # Insert marker kwarg just before the closing paren
        before = code[:insertion].rstrip()
        after = code[insertion:]
        separator = ", " if before and before[-1] not in ('(', ' ') else ""
        return f"{before}{separator}marker='{marker}'{after}"
    return code


def replace_linewidth(code: str) -> str:
    linewidth = random.choice(LINEWIDTHS)
    if "linewidth=" in code:
        return re.sub(r"linewidth\s*=\s*[\d.]+", f"linewidth={linewidth}", code)
    if "lw=" in code:
        return re.sub(r"lw\s*=\s*[\d.]+", f"lw={linewidth}", code)
    return code


def replace_alpha(code: str) -> str:
    alpha = random.choice(ALPHAS)
    if "alpha=" in code:
        return re.sub(r"alpha\s*=\s*[\d.]+", f"alpha={alpha}", code)
    return code


def replace_linestyle(code: str) -> str:
    linestyle = random.choice(LINESTYLES)
    if "linestyle=" in code:
        return re.sub(r"linestyle\s*=\s*['\"].*?['\"]", f"linestyle='{linestyle}'", code)
    if "ls=" in code:
        return re.sub(r"ls\s*=\s*['\"].*?['\"]", f"ls='{linestyle}'", code)
    # Use AST to find the actual plot call and inject linestyle= safely
    insertion = _find_safe_plot_call_insertion_point(code)
    if insertion is not None:
        before = code[:insertion].rstrip()
        after = code[insertion:]
        separator = ", " if before and before[-1] not in ('(', ' ') else ""
        return f"{before}{separator}linestyle='{linestyle}'{after}"
    return code


def change_cmap(code: str) -> str:
    if "cmap=" not in code:
        return code
    cmap = random.choice(COLORMAPS)
    return re.sub(r"cmap\s*=\s*['\"].*?['\"]", f"cmap='{cmap}'", code)


def change_figsize(code: str) -> str:
    size = random.choice([(5, 4), (6, 4), (7, 5), (8, 5), (9, 6)])
    if "figsize=" in code:
        return re.sub(r"figsize\s*=\s*\([^\)]*\)", f"figsize={size}", code)
    if "plt.figure(" in code:
        return re.sub(r"plt\.figure\((.*?)\)", lambda m: f"plt.figure({append_kwarg(m.group(1), f'figsize={size}')})", code, count=1)
    if "plt.subplots(" in code:
        return re.sub(r"plt\.subplots\((.*?)\)", lambda m: f"plt.subplots({append_kwarg(m.group(1), f'figsize={size}')})", code, count=1)
    return code


def add_grid(code: str) -> str:
    if "imshow(" in code or "heatmap(" in code:
        return code
    # Only match .grid( calls (ax.grid / plt.grid), NOT meshgrid(
    if re.search(r'\.grid\s*\(', code):
        return re.sub(r'(?<=\.)(grid\s*\([^\)]*\))', "grid(True, linestyle='--', alpha=0.35)", code, count=1)
    if "sns.set_theme(" in code:
        return code
    return code.rstrip() + "\nplt.grid(True, linestyle='--', alpha=0.35)\n"


def change_style(code: str) -> str:
    if "sns." in code:
        style = random.choice(SEABORN_STYLES)
        if "sns.set_theme(" in code:
            return re.sub(
                r"sns\.set_theme\((.*?)\)",
                lambda m: f"sns.set_theme({append_or_replace_kwarg(m.group(1), 'style', repr(style))})",
                code,
                count=1,
            )
        if "sns.set_style(" in code:
            return re.sub(r"sns\.set_style\(\s*['\"].*?['\"]\s*\)", f"sns.set_style('{style}')", code, count=1)
        return "import seaborn as sns\nsns.set_theme(style='{}')\n{}".format(style, code)

    style = random.choice(MATPLOTLIB_STYLES)
    if "plt.style.use(" in code:
        return re.sub(r"plt\.style\.use\(\s*['\"].*?['\"]\s*\)", f"plt.style.use('{style}')", code, count=1)
    return "import matplotlib.pyplot as plt\nplt.style.use('{}')\n{}".format(style, code)


def replace_seed(code: str) -> str:
    seed = random.randint(0, 10000)
    if "np.random.seed(" in code:
        return re.sub(r"np\.random\.seed\(\d+\)", f"np.random.seed({seed})", code, count=1)
    if "RandomState(" in code:
        return re.sub(r"RandomState\(\d+\)", f"RandomState({seed})", code, count=1)
    return code


def change_bins(code: str) -> str:
    if "hist(" not in code or "bins=" not in code:
        return code
    bins = random.choice([10, 15, 20, 24, 30, 36])
    return re.sub(r"bins\s*=\s*\d+", f"bins={bins}", code)


def append_kwarg(arg_text: str, new_kwarg: str) -> str:
    arg_text = arg_text.strip()
    if not arg_text:
        return new_kwarg
    return f"{arg_text}, {new_kwarg}"


def append_or_replace_kwarg(arg_text: str, key: str, value_literal: str) -> str:
    pattern = rf"{key}\s*=\s*[^,]+"
    if re.search(pattern, arg_text):
        return re.sub(pattern, f"{key}={value_literal}", arg_text, count=1)
    return append_kwarg(arg_text, f"{key}={value_literal}")


def fix_invalid_colormaps(code: str) -> str:
    return re.sub(r"['\"]petroff\d+['\"]", "'viridis'", code)


def fix_invalid_params(code: str) -> str:
    def clamp_alpha(match):
        try:
            value = min(max(float(match.group(1)), 0.0), 1.0)
            return f"alpha={value}"
        except Exception:
            return match.group(0)

    return re.sub(r"alpha\s*=\s*([0-9.]+)", clamp_alpha, code)


def fix_invalid_legend_loc(code: str) -> str:
    """Fix invalid loc= values, but only in legend() calls.

    'best' is only valid for ax.legend() / plt.legend(), not for
    fig.legend(), set_xlabel(loc=...), AnchoredText(loc=...), or
    new_fixed_axis(loc=...). We therefore only replace loc= values
    that appear within a .legend( context.
    """
    legend_valid_locs = {
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    }

    # Only match loc= that is preceded by .legend( on the same logical call
    # Pattern: .legend( ... loc='value' ... )
    def replace_legend_loc(match):
        full = match.group(0)
        loc_value = match.group(2)
        if loc_value not in legend_valid_locs:
            return full.replace(f"loc='{loc_value}'", "loc='upper right'").replace(
                f'loc="{loc_value}"', "loc='upper right'"
            )
        return full

    # Match .legend( ... loc='...' ... ) — handles multi-arg calls
    code = re.sub(
        r"(\.legend\s*\([^)]*?)loc\s*=\s*['\"]([^'\"]*?)['\"]([^)]*?\))",
        replace_legend_loc,
        code,
    )

    # Also fix fig.legend(loc='best') which is invalid for figure-level legend
    def fix_fig_legend_loc(match):
        loc_value = match.group(2)
        if loc_value == "best":
            return match.group(0).replace("loc='best'", "loc='upper right'").replace(
                'loc="best"', "loc='upper right'"
            )
        return match.group(0)

    code = re.sub(
        r"(fig\.legend\s*\([^)]*?)loc\s*=\s*['\"]([^'\"]*?)['\"]([^)]*?\))",
        fix_fig_legend_loc,
        code,
    )

    return code


def remove_unsupported_kwargs(code: str) -> str:
    code = re.sub(r",?\s*hatchcolor\s*=\s*[^,\)\n]+", "", code)
    code = re.sub(r",?\s*over\s*=\s*[^,\)\n]+", "", code)
    return code


def semantic_transform_pool(code: str):
    transforms = [change_figsize, change_style, replace_alpha, replace_seed]
    if "color=" in code or "c=" in code or "facecolor=" in code or "edgecolor=" in code:
        transforms.append(replace_palette)
    if "plot(" in code or "lineplot(" in code:
        transforms.extend([replace_marker, replace_linewidth, replace_linestyle])
    if "hist(" in code or "histplot(" in code:
        transforms.append(change_bins)
    if "cmap=" in code or "imshow(" in code or "heatmap(" in code:
        transforms.append(change_cmap)
    if "scatter(" in code or "bar(" in code:
        transforms.append(replace_alpha)
    if "imshow(" not in code and "heatmap(" not in code:
        transforms.append(add_grid)
    return transforms


def build_augmented_code(original_code: str) -> str | None:
    for _ in range(4):
        new_code = deepcopy(original_code)
        original_colors = extract_colors(original_code)
        mappings = generate_color_mappings(original_colors, num_variants=1)
        if mappings:
            new_code = apply_color_mapping(new_code, mappings[0])

        transforms = semantic_transform_pool(new_code)
        random.shuffle(transforms)
        changed = False

        for transform in transforms[: random.randint(3, min(5, len(transforms)))]:
            try:
                candidate = transform(new_code)
            except Exception:
                continue
            if candidate != new_code:
                changed = True
                new_code = candidate

        if not changed:
            continue

        new_code = fix_invalid_colormaps(new_code)
        new_code = fix_invalid_params(new_code)
        new_code = remove_unsupported_kwargs(new_code)
        new_code = fix_invalid_legend_loc(new_code)
        new_code = sanitize_code_for_execution(new_code)

        if is_valid_python(new_code):
            return new_code

    return None


def augment_file(py_path, variant_idx, AUG_CODE_DIR, GALLERY_DIR):
    try:
        with open(py_path, "r", encoding="utf-8", errors="ignore") as handle:
            original_code = handle.read()
    except Exception:
        return None

    if should_skip_code(original_code):
        return None

    base = Path(py_path).stem
    original_img = os.path.join(GALLERY_DIR, f"{base}.png")
    original_signature = compute_image_signature(original_img) if os.path.exists(original_img) else None

    outputs = []
    local_image_signatures = set()
    local_code_signatures = set()

    for idx in range(3):
        out_py = os.path.join(AUG_CODE_DIR, f"{base}_aug_{variant_idx}_{idx}.py")
        out_img = os.path.join(GALLERY_DIR, f"{base}_aug_{variant_idx}_{idx}.png")

        new_code = build_augmented_code(original_code)
        if not new_code:
            continue

        current_code_signature = code_fingerprint(new_code)
        if current_code_signature in local_code_signatures:
            continue

        try:
            with open(out_py, "w", encoding="utf-8") as handle:
                handle.write(new_code)
        except Exception:
            continue

        success = render_code_to_png(new_code, out_img, desc="augment")
        if not success or not os.path.exists(out_img):
            try:
                os.remove(out_py)
            except OSError:
                pass
            continue

        if is_low_information_image(out_img):
            try:
                os.remove(out_img)
                os.remove(out_py)
            except OSError:
                pass
            continue

        image_signature = compute_image_signature(out_img)
        if image_signature is None:
            try:
                os.remove(out_img)
                os.remove(out_py)
            except OSError:
                pass
            continue

        if image_signature == original_signature:
            try:
                os.remove(out_img)
                os.remove(out_py)
            except OSError:
                pass
            continue

        if image_signature in local_image_signatures:
            try:
                os.remove(out_img)
                os.remove(out_py)
            except OSError:
                pass
            continue

        if os.path.exists(original_img) and not is_different(out_img, original_img):
            try:
                os.remove(out_img)
                os.remove(out_py)
            except OSError:
                pass
            continue

        local_image_signatures.add(image_signature)
        local_code_signatures.add(current_code_signature)
        outputs.append({"image": out_img, "code": new_code})

    return outputs or None


def worker(worker_id, input_q, output_q, AUG_CODE_DIR, GALLERY_DIR):
    while True:
        assignment = input_q.get()
        if assignment is None:
            output_q.put(("stopped", worker_id, None, None, None))
            break

        assignment_id, task = assignment
        output_q.put(("started", worker_id, assignment_id, task, None))
        try:
            result = augment_file(task[0], task[1], AUG_CODE_DIR, GALLERY_DIR)
            output_q.put(("done", worker_id, assignment_id, task, result))
        except Exception as exc:
            output_q.put(("error", worker_id, assignment_id, task, str(exc)))


def terminate_process(process, timeout=2):
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout=timeout)


def close_queue(queue):
    try:
        queue.cancel_join_thread()
    except Exception:
        pass
    try:
        queue.close()
    except Exception:
        pass
    try:
        queue.join_thread()
    except Exception:
        pass


def augment_dataset(GALLERY_DIR, train_dataset_folder, AUG_CODE_DIR, JSONL_PATH):
    os.makedirs(AUG_CODE_DIR, exist_ok=True)
    os.makedirs(GALLERY_DIR, exist_ok=True)

    py_files = get_all_py_files(train_dataset_folder)
    all_tasks = [(str(py_file), variant_idx) for py_file in py_files for variant_idx in range(VARIANTS_PER_FILE)]
    total_batches = (len(all_tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    total_results = 0
    seen_image_signatures = set()
    seen_code_signatures = set()

    with open(JSONL_PATH, "a", encoding="utf-8") as handle:
        for batch_index, batch_start in enumerate(range(0, len(all_tasks), BATCH_SIZE), start=1):
            batch_tasks = all_tasks[batch_start : batch_start + BATCH_SIZE]
            task_queue = deque(batch_tasks)
            retry_count = {}

            ctx = mp.get_context("spawn")
            output_q = ctx.Queue()
            workers = {}
            input_queues = {}
            worker_tasks = {}
            next_assignment_id = 0

            def start_worker(worker_id: int):
                old_queue = input_queues.pop(worker_id, None)
                if old_queue is not None:
                    close_queue(old_queue)

                input_q = ctx.Queue()
                process = ctx.Process(
                    target=worker,
                    args=(worker_id, input_q, output_q, AUG_CODE_DIR, GALLERY_DIR),
                )
                process.daemon = True
                process.start()
                input_queues[worker_id] = input_q
                workers[worker_id] = process
                worker_tasks[worker_id] = None

            def stop_worker(worker_id: int):
                process = workers.get(worker_id)
                if process is not None:
                    terminate_process(process)

                queue = input_queues.pop(worker_id, None)
                if queue is not None:
                    close_queue(queue)

            def is_current_event(worker_id: int, assignment_id: int, task) -> bool:
                current = worker_tasks.get(worker_id)
                return (
                    current is not None
                    and current["assignment_id"] == assignment_id
                    and current["task"] == task
                )

            def assign_task(worker_id: int):
                nonlocal next_assignment_id
                if worker_tasks.get(worker_id) is not None or not task_queue:
                    return

                task = task_queue.popleft()
                next_assignment_id += 1
                assignment_id = next_assignment_id
                worker_tasks[worker_id] = {
                    "assignment_id": assignment_id,
                    "task": task,
                    "assigned_at": time.time(),
                    "started_at": None,
                }
                input_queues[worker_id].put((assignment_id, task))

            def requeue_or_finish(task, retry_limit):
                nonlocal completed
                retries = retry_count.get(task, 0) + 1
                retry_count[task] = retries
                if retries <= retry_limit:
                    task_queue.append(task)
                else:
                    completed += 1
                    pbar.update(1)

            print(f"\nProcessing batch {batch_index}/{total_batches}")

            worker_count = min(max(1, MAX_WORKERS), max(1, len(batch_tasks)))
            for worker_id in range(worker_count):
                start_worker(worker_id)

            for worker_id in list(workers):
                assign_task(worker_id)

            completed = 0
            pbar = tqdm(total=len(batch_tasks), desc="Batch progress")
            last_heartbeat = time.time()

            while completed < len(batch_tasks):
                try:
                    event, worker_id, assignment_id, task, payload = output_q.get(timeout=1)
                except Exception:
                    event = None

                if event == "started":
                    if is_current_event(worker_id, assignment_id, task):
                        worker_tasks[worker_id]["started_at"] = time.time()
                elif event == "done":
                    if not is_current_event(worker_id, assignment_id, task):
                        continue

                    worker_tasks[worker_id] = None
                    completed += 1
                    pbar.update(1)

                    results = payload or []
                    for sample in results:
                        image_signature = compute_image_signature(sample["image"])
                        code_signature = code_fingerprint(sample["code"])

                        if not image_signature:
                            continue
                        if image_signature in seen_image_signatures:
                            continue
                        if code_signature in seen_code_signatures:
                            continue

                        seen_image_signatures.add(image_signature)
                        seen_code_signatures.add(code_signature)
                        handle.write(json.dumps(sample) + "\n")
                        handle.flush()
                        total_results += 1

                    assign_task(worker_id)
                elif event == "error":
                    if not is_current_event(worker_id, assignment_id, task):
                        continue

                    worker_tasks[worker_id] = None
                    requeue_or_finish(task, MAX_RETRIES)
                    stop_worker(worker_id)
                    start_worker(worker_id)
                    assign_task(worker_id)

                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    for worker_id, process in list(workers.items()):
                        current = worker_tasks.get(worker_id)

                        if not process.is_alive():
                            if current is not None:
                                worker_tasks[worker_id] = None
                                requeue_or_finish(current["task"], MAX_RETRIES)
                            stop_worker(worker_id)
                            start_worker(worker_id)
                            assign_task(worker_id)
                            continue

                        if current is None:
                            assign_task(worker_id)
                            continue

                        started_at = current["started_at"] or current["assigned_at"]
                        if now - started_at > TIMEOUT:
                            worker_tasks[worker_id] = None
                            requeue_or_finish(current["task"], MAX_TIMEOUT_RETRIES)
                            stop_worker(worker_id)
                            start_worker(worker_id)
                            assign_task(worker_id)

                    last_heartbeat = now

                if not task_queue and not any(worker_tasks.values()) and completed >= len(batch_tasks):
                    break

            pbar.close()

            for worker_id, process in list(workers.items()):
                if process.is_alive():
                    try:
                        input_queues[worker_id].put(None)
                    except Exception:
                        pass

            for process in workers.values():
                if process.is_alive():
                    process.join(timeout=2)
                terminate_process(process)

            for queue in input_queues.values():
                close_queue(queue)
            close_queue(output_q)

    print(f"\nTotal generated samples: {total_results}")
