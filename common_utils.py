import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

# ── Disable TorchDynamo / torch.compile BEFORE importing torch ──────────────
# Unsloth compiles Qwen3-VL vision blocks, while device_map="auto" installs
# Accelerate AlignDevicesHook (marked @torch.compiler.disable).  That
# combination causes a fatal graph-break error during model.generate().
# Setting these env vars before torch is imported fully prevents the conflict.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import matplotlib.pyplot as plt
import torch
import gc
import warnings, json, ast, matplotlib as mpl

warnings.filterwarnings("ignore")
mpl.use("Agg")

import re
from pathlib import Path

root_dir = Path().resolve()
import io
import tokenize
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from skimage.metrics import structural_similarity as ssim
from IPython.display import display, Markdown
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from vegalite_utils import (
    VEGA_LITE_INSTRUCTION,
    UNIFIED_INSTRUCTION,
    canonical_json_dumps,
    evaluate_vegalite_ir,
    parse_vegalite_spec,
    vegalite_to_matplotlib,
)

import builtins
from threading import Lock

error_lock = Lock()
ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|model)\s*\n", flags=re.IGNORECASE)

try:
    import torch._dynamo as torch_dynamo
except Exception:
    torch_dynamo = None

train_gallery_dir = root_dir / "plot_gallery_train"
train_dataset_file = root_dir / "plot2code_train.jsonl"

test_gallery_dir = root_dir / "plot_gallery_test"
test_dataset_file = root_dir / "plot2code_test.jsonl"

# ----------------------------------------
# Memory Management
# ----------------------------------------


def cleanup_memory(clear_gpu=True):
    """Clear CPU and GPU memory caches."""
    gc.collect()
    if clear_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("✓ Memory cleaned")


def configure_eager_generation():
    """Keep inference generation out of torch.compile/Dynamo graph capture.

    Unsloth can compile Qwen-VL vision blocks, while device_map="auto" adds
    Accelerate AlignDevicesHook callbacks that are marked torch.compiler.disable.
    That combination raises a fatal graph-break error during generate().
    We fully disable Dynamo for reliable eager-mode inference.
    """
    # Environment-level disable (idempotent, catches any late imports)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    if torch_dynamo is None:
        return
    try:
        torch_dynamo.config.suppress_errors = True
        # Disable Dynamo globally — prevents compiled submodule forwards
        torch_dynamo.disable(recursive=True)
        torch_dynamo.reset()
    except Exception:
        pass


def remove_accelerate_hooks(model):
    """Remove Accelerate dispatch hooks that conflict with compiled graphs.

    Safe to call even when no hooks are attached.  Should only be called
    when the model fits on a single device (all parameters already placed).
    """
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(model)
    except Exception:
        pass


def eager_generate(model, inputs, generation_kwargs):
    """Run model.generate with Dynamo fully disabled."""
    configure_eager_generation()
    return model.generate(**inputs, **generation_kwargs)


def remove_comments(code):
    """
    Safely remove comments without:
    - breaking string literals (e.g., "#FFAA00")
    - breaking indentation (VERY IMPORTANT)
    """

    # -------------------------
    # 1. Remove comments safely using tokenize
    # -------------------------
    try:
        io_obj = io.StringIO(code)
        output_tokens = []

        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok.type
            token_string = tok.string

            # Skip ONLY comments
            if token_type == tokenize.COMMENT:
                continue

            output_tokens.append(tok)

        code = tokenize.untokenize(output_tokens)

    except Exception:
        pass  # fallback if something weird happens

    # -------------------------
    # 2. Remove standalone docstrings ONLY (not affecting indentation)
    # -------------------------
    code = re.sub(
        r'^\s*(("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))', "", code, flags=re.MULTILINE
    )

    # -------------------------
    # 3. Clean extra empty lines ONLY (do NOT strip indentation)
    # -------------------------
    code = re.sub(r"\n\s*\n+", "\n\n", code)

    return code


def check_png_quality(image_name, out_path):
    # -------------------------
    # 2. FILE EXISTS CHECK
    # -------------------------
    if not os.path.exists(out_path):
        print(f"Image not saved for: {image_name}")
        return False

    # -------------------------
    # 3. CORRUPTION CHECK
    # -------------------------
    try:
        with Image.open(out_path) as img:
            img.verify()
    except Exception as e:
        os.remove(out_path)
        print(f"Corrupted image: {image_name}")
        return False

    # reopen after verify
    img = Image.open(out_path).convert("L")
    arr = np.array(img)

    # -------------------------
    # 4. SIZE CHECK
    # -------------------------
    if arr.shape[0] < 50 or arr.shape[1] < 50:
        os.remove(out_path)
        print(f"Image too small for: {image_name} (size: {arr.shape})")
        return False

    # -------------------------
    # 5. BLANK CHECK (bbox)
    # -------------------------
    if img.getbbox() is None:
        os.remove(out_path)
        print(f"Blank image detected for: {image_name}")
        return False

    # -------------------------
    # 6. NEAR-BLANK CHECK (mean intensity)
    # -------------------------
    # if arr.mean() > 245:
    #     os.remove(out_path)
    #     print(
    #         f"Near-blank (too white) image detected for: {image_name} (mean intensity: {arr.mean():.1f})"
    #     )
    #     return False

    # -------------------------
    # 7. LOW VARIANCE CHECK
    # -------------------------
    # if arr.std() < 5:
    #     os.remove(out_path)
    #     print(f"Low variance image detected for: {image_name} (std: {arr.std():.1f})")
    #     return False

    # -------------------------
    # 8. EDGE CONTENT CHECK (optional but useful)
    # -------------------------
    # edges = np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()
    # if edges < 2:
    #     os.remove(out_path)
    #     print(
    #         f"No structure (flat image) detected for: {image_name} (edges: {edges:.1f})"
    #     )
    #     return False

    return True


# Check if png file exists at the path or not
def check_png_quality_from_json(jsonl_file):
    count_corr = 0
    count_good = 0
    to_delete = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_path = data.get("image")

            try:
                image_name = image_path.split("/")[-1]
                if check_png_quality(image_name, image_path):
                    count_good += 1
                else:
                    count_corr += 1
                    to_delete.append(image_path)
                    print(f"Corrupted or missing file: {image_path}")
            except Exception as e:
                count_corr += 1
                to_delete.append(image_path)
                print(f"Error occurred while checking {image_path}: {e}")

    print(f"Total PNG files checked: {count_good + count_corr}")
    if count_corr > 0:
        print(f"Total corrupted PNG files: {count_corr}")
        # Delete corrupted files and remove entries from JSONL if needed
        for img_path in to_delete:
            try:
                if os.path.isfile(img_path):
                    os.remove(img_path)
                    print(f"Deleted corrupted/missing file: {img_path}")
            except Exception as e:
                print(f"Error deleting file {img_path}: {e}")

        # Optionally, you can also remove entries from the JSONL file if needed
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for line in lines:
                data = json.loads(line)
                if data.get("image") not in to_delete:
                    f.write(line)
    else:
        print(f"{count_good} PNG files exist.")


# ----------------------------------------
# Code Execution
# ----------------------------------------


def render_code_to_png(py_code, out_path, desc=""):
    """Execute matplotlib code → save PNG → validate image quality"""

    old_argv = sys.argv[:]
    old_parser = argparse.ArgumentParser

    try:
        # 🔥 FIX: Remove Jupyter args (argparse bug)
        sys.argv = [sys.argv[0]]

        # 🔥 FIX: Safe argparse override
        class SafeArgumentParser(argparse.ArgumentParser):
            def parse_args(self, *args, **kwargs):
                return super().parse_known_args(*args, **kwargs)[0]

        argparse.ArgumentParser = SafeArgumentParser

        safe_builtins = {
            "range": range,
            "len": len,
            "print": print,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "isinstance": isinstance,
            "ord": ord,
            "chr": chr,
            "any": any,
            "all": all,
            "next": next,
            "slice": slice,
            "pow": pow,
            "callable": callable,
            "getattr": getattr,
            "hasattr": hasattr,
            "setattr": setattr,
            "issubclass": issubclass,
            "iter": iter,
            "id": id,
            "format": format,
            "hash": hash,
            "vars": vars,
            "dir": dir,
            "eval": eval,
            "exec": exec,
            "super": super,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "property": property,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            "memoryview": memoryview,
            # Exception types
            "Exception": Exception,
            "BaseException": BaseException,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            "NotImplementedError": NotImplementedError,
            "ZeroDivisionError": ZeroDivisionError,
            "OverflowError": OverflowError,
            "NameError": NameError,
            "IOError": IOError,
            "OSError": OSError,
            "FileNotFoundError": FileNotFoundError,
            "ImportError": ImportError,
            "ModuleNotFoundError": ModuleNotFoundError,
            "ArithmeticError": ArithmeticError,
            "LookupError": LookupError,
            "UnicodeError": UnicodeError,
            "UnicodeDecodeError": UnicodeDecodeError,
            "UnicodeEncodeError": UnicodeEncodeError,
            # Core
            "__import__": builtins.__import__,
            "__name__": "__main__",
            "__build_class__": builtins.__build_class__,
            "complex": complex,
            "divmod": divmod,
            "repr": repr,
            "type": type,
            "object": object,
            "mpl": __import__("matplotlib"),
        }

        safe_globals = {
            "__builtins__": safe_builtins,
            "np": __import__("numpy"),
            "pd": pd,
            "plt": plt,
            "matplotlib": __import__("matplotlib"),
            "json": json,
        }

        py_code = py_code.replace("plt.show()", "")

        original_register = mpl.colormaps.register
        mpl.colormaps.register = lambda *a, **kw: original_register(
            *a, **{**kw, "force": True}
        )

        plt.close("all")

        # -------------------------
        # 1. EXECUTE CODE
        # -------------------------
        exec(py_code, safe_globals, None)
        image_name = out_path.split("/")[-1]

        fig_nums = plt.get_fignums()
        if not fig_nums:
            print(f"No figure created for image: {image_name}")
            return False

        fig = plt.figure(fig_nums[-1])

        if len(fig.axes) == 0:
            print(f"No axes in figure for image: {image_name}")
            return False

        # 🔥 Ensure something is plotted
        has_content = False
        for ax in fig.axes:
            if (
                ax.has_data()
                or len(ax.lines) > 0
                or len(ax.patches) > 0
                or len(ax.collections) > 0
                or len(ax.images) > 0
                or len(ax.texts) > 0
            ):
                has_content = True
                break

        if not has_content:
            print(f"\nNo plotted content for image: {image_name}")
            return False

        fig.canvas.draw()

        fig.savefig(out_path, bbox_inches="tight", dpi=min(fig.dpi, 150))
        plt.close(fig)

        return check_png_quality(image_name, out_path)

    except Exception as e:
        error_file = f"{desc}_errors.jsonl"
        existing_errors = set()

        with error_lock:
            if Path(error_file).exists():
                with open(error_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            err_obj = json.loads(line)
                            existing_errors.add(err_obj.get("Error"))
                        except:
                            continue

            error_str = str(e)
            if error_str not in existing_errors:
                with open(error_file, "a", encoding="utf-8") as f:
                    json.dump({"Code": py_code, "Error": error_str}, f)
                    f.write("\n")

        return False

    finally:
        argparse.ArgumentParser = old_parser
        sys.argv = old_argv
        mpl.colormaps.register = original_register
        plt.close("all")


# ----------------------------------------
# Code Extraction & Metrics
# ----------------------------------------


def extract_code(text):
    """Extract fenced Python code from LLM output."""
    text = ROLE_PREFIX_RE.sub("", (text or "").strip())
    matches = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if matches:
        return matches[-1].strip()  # Take the last code block found
    return text.strip()


def repair_truncated_code(code, min_fraction=0.6):
    """
    Recover from truncated generations by dropping incomplete trailing lines until
    the snippet becomes parseable. This turns many syntax failures into runnable code.
    """
    code = (code or "").strip()
    if not code:
        return code

    try:
        ast.parse(code)
        return code
    except Exception:
        pass

    lines = [
        line
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    if not lines:
        return code

    minimum_lines = max(1, int(len(lines) * min_fraction))
    for end in range(len(lines) - 1, minimum_lines - 1, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except Exception:
            continue

    return code


def sanitize_generated_code(text, instruction=VEGA_LITE_INSTRUCTION):
    """Normalize model output before execution and metric scoring.

    Vega task: image -> Vega-Lite JSON -> Matplotlib code.
    Legacy task: image -> Python/Matplotlib code directly.
    """
    return sanitize_generated_output(text, instruction)


def sample_instruction(sample):
    """Return the task instruction implied by the sample's explicit vega_spec field."""
    spec = parse_vegalite_spec(sample.get("vega_spec"))
    if spec is not None:
        return VEGA_LITE_INSTRUCTION
    return UNIFIED_INSTRUCTION


def explicit_vegalite_spec(sample):
    """Return a spec only when the explicit vega_spec field is nonempty/parseable."""
    return parse_vegalite_spec(sample.get("vega_spec"))


def sanitize_generated_output(text, instruction):
    """Normalize generated output according to the task instruction."""
    if instruction != VEGA_LITE_INSTRUCTION:
        return repair_truncated_code(extract_code(text))

    spec = parse_vegalite_spec(text)
    if spec is None:
        return "raise RuntimeError('Model output was not valid Vega-Lite JSON')"
    try:
        return vegalite_to_matplotlib(spec)
    except Exception as exc:
        return f"raise RuntimeError({str(exc)!r})"


def reference_code_from_sample(sample):
    """Return executable Matplotlib reference code for either task type."""
    spec = explicit_vegalite_spec(sample)
    if spec is not None:
        return vegalite_to_matplotlib(spec)

    code = sample.get("code")
    if code:
        return code
    raise KeyError("sample must contain either a nonempty vega_spec or code")


def extract_calls(py_code):
    """Return list of Matplotlib function calls used in code."""
    try:
        tree = ast.parse(py_code)
    except Exception:
        return []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    return calls


def image_ssim(img_a, img_b, max_size=512):
    a = Image.open(img_a).convert("L")
    b = Image.open(img_b).convert("L")

    a.thumbnail((max_size, max_size))
    b.thumbnail((max_size, max_size))

    a = np.array(a)
    b = np.array(b)

    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return float(ssim(a[:h, :w], b[:h, :w]))


def code_bleu(ref, gen):
    """Compute smoothed BLEU for short code snippets."""
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)


def code_edit_sim(ref, gen):
    """Edit-distance-based code similarity."""
    return SequenceMatcher(None, ref, gen).ratio()


def api_overlap_verbose(ref_code, gen_code):
    """Compute API overlap and return matched/missed/extra calls."""
    ref_calls = set(extract_calls(ref_code))
    gen_calls = set(extract_calls(gen_code))
    matched = ref_calls & gen_calls
    missed = ref_calls - gen_calls
    extras = gen_calls - ref_calls
    sim = len(matched) / len(ref_calls) if ref_calls else 0.0
    return sim, matched, missed, extras

# ----------------------------------------
# Code Generation
# ----------------------------------------

def gen_code_for_image_batch(
    model,
    processor,
    img_paths,
    max_new_tokens=1200,
    temperature=0.0,
    instructions=None,
    samples=None,
):
    """
    Generate model outputs for a batch of images using per-sample task instructions.

    Args:
        model: Vision-language model
        processor: Model processor/tokenizer
        img_paths: List of image file paths
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        instructions: Optional per-image instruction list.
        samples: Optional per-image sample dicts; used to derive instructions from vega_spec.

    Returns:
        List of raw generated text strings (one per image), or [] on failure
    """

    if instructions is None and samples is not None:
        instructions = [sample_instruction(sample) for sample in samples]
    elif instructions is None:
        instructions = [VEGA_LITE_INSTRUCTION for _ in img_paths]
    elif isinstance(instructions, str):
        instructions = [instructions for _ in img_paths]
    if len(instructions) != len(img_paths):
        raise ValueError("instructions must be None, a string, or one instruction per image")

    # Build per-image instructions.
    images, prompt_instructions = [], []
    for img_path, base_instruction in zip(img_paths, instructions):
        with Image.open(img_path) as img:
            images.append(img.convert("RGB").copy())

        prompt_instructions.append(base_instruction)

    # Build batch of chat-formatted prompts
    batch_convos = []
    for img, instruction in zip(images, prompt_instructions):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        convo = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch_convos.append(convo)

    inputs = processor(
        text=batch_convos,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    # Retry on GPU OOM or generation errors (not the same as code execution errors)
    max_gpu_attempts = 3
    for attempt in range(1, max_gpu_attempts + 1):
        try:
            with torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "use_cache": True,
                }
                if temperature > 0:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = 0.9

                outputs = eager_generate(model, inputs, generation_kwargs)
            # results = processor.batch_decode(outputs, skip_special_tokens=True)
            generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            results = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            # results = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            del images, prompt_instructions, batch_convos, inputs, outputs
            cleanup_memory(clear_gpu=True)
            return results

        except Exception as e:
            print(f"GPU/generation error (attempt {attempt}/{max_gpu_attempts}): {e}")
            cleanup_memory(clear_gpu=True)

    return []


def generate_code_batch_with_fallback(
    model,
    processor,
    img_paths,
    max_new_tokens=1200,
    temperature=0.0,
    instructions=None,
    samples=None,
):
    """
    Try batched generation first, then fall back to single-image generation if the
    whole batch fails. This avoids dropping good examples because one large batch OOMed.
    """
    batch_codes = gen_code_for_image_batch(
        model,
        processor,
        img_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        instructions=instructions,
        samples=samples,
    )
    if batch_codes and len(batch_codes) == len(img_paths):
        return batch_codes

    recovered = []
    if instructions is None and samples is not None:
        instructions = [sample_instruction(sample) for sample in samples]
    elif instructions is None:
        instructions = [VEGA_LITE_INSTRUCTION for _ in img_paths]
    elif isinstance(instructions, str):
        instructions = [instructions for _ in img_paths]
    for img_path, instruction in zip(img_paths, instructions):
        single = gen_code_for_image_batch(
            model,
            processor,
            [img_path],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            instructions=[instruction],
        )
        recovered.append(single[0] if single else "")
    return recovered


# ----------------------------------------
# Evaluation
# ----------------------------------------


def evaluate_single_example(ref_data, result_text, op_path="test", instruction=None):
    """
    Evaluate one generated result against the reference using the sample's task type.

    Args:
        model: Vision-language model
        processor: Model processor/tokenizer
        ref_data: Dict with 'code' or Vega-Lite spec, plus 'image'
        result_text: Initial generated text from the batch pass
        op_path: Output directory for rendered images

    Returns:
        Dict with evaluation metrics and execution_success flag
    """
    instruction = instruction or sample_instruction(ref_data)
    is_vega_task = instruction == VEGA_LITE_INSTRUCTION
    ref_spec = explicit_vegalite_spec(ref_data) if is_vega_task else None
    ref_code = reference_code_from_sample(ref_data)
    ref_image = ref_data["image"]
    generated_spec = parse_vegalite_spec(result_text) if is_vega_task else None
    ir_scores = (
        evaluate_vegalite_ir(ref_spec, generated_spec)
        if ref_spec is not None and generated_spec is not None
        else {
            "exact_match": False,
            "field_precision": 0,
            "field_recall": 0,
            "field_f1": 0,
            "tree_similarity": 0,
        }
    )

    op_path = f"./{op_path}"

    print(f"\nEvaluating: {ref_image}")
    print("=" * 80)

    # --- Try the initial result from the batch generation pass ---
    os.makedirs(op_path, exist_ok=True)
    gen_path = f"{op_path}/{os.path.basename(ref_image)}"
    code = sanitize_generated_output(result_text, instruction)
    ok = render_code_to_png(code, gen_path, desc=op_path.split("/")[-1][:10])

    # --- Return zero-score result on total failure ---
    if not ok:
        return {
            "image": ref_image,
            "ssim": 0,
            "bleu": 0,
            "edit_sim": 0,
            "api_sim": 0,
            "code_score": 0,
            "overall": 0,
            "execution_success": False,
            "instruction": instruction,
            "generated_vega_lite_spec": generated_spec,
            "generated_code": code,
            "ir_exact_match": ir_scores["exact_match"],
            "ir_field_precision": ir_scores["field_precision"],
            "ir_field_recall": ir_scores["field_recall"],
            "ir_field_f1": ir_scores["field_f1"],
            "ir_tree_similarity": ir_scores["tree_similarity"],
        }

    # --- Log the successful code ---
    print(f"Image saved at {gen_path}")
    print("-" * 80)

    # --- Compute metrics ---
    ssim_score = image_ssim(ref_image, gen_path)
    bleu = code_bleu(ref_code, code)
    edit_sim = code_edit_sim(ref_code, code)
    api_sim, matched, missed, extras = api_overlap_verbose(ref_code, code)
    code_score = 0.4 * bleu + 0.3 * edit_sim + 0.3 * api_sim
    overall = 0.5 * ssim_score + 0.5 * code_score

    # --- Log API details ---
    print("\nAPI Usage Analysis")
    print("-" * 80)
    print(
        f"Reference APIs ({len(matched)+len(missed)}): {sorted(extract_calls(ref_code))}"
    )
    print(f"Generated APIs ({len(matched)+len(extras)}): {sorted(extract_calls(code))}")
    print(f"Matched:  {sorted(matched)}")
    print(f"Missed:   {sorted(missed)}")
    print(f"Extra:    {sorted(extras)}")

    # --- Log scores ---
    print("\nQuantitative Scores")
    print("-" * 80)
    print(f"SSIM (Visual):       {ssim_score:.3f}")
    print(f"BLEU (Text):         {bleu:.3f}")
    print(f"Edit Similarity:     {edit_sim:.3f}")
    print(f"API Overlap:         {api_sim:.3f}")
    print(f"Composite CodeScore: {code_score:.3f}")
    print(f"Overall Score:       {overall:.3f}")
    print(f"IR Exact Match:      {ir_scores['exact_match']}")
    print(f"IR Field F1:         {ir_scores['field_f1']:.3f}")
    print(f"IR Tree Similarity:  {ir_scores['tree_similarity']:.3f}")
    print("=" * 80)

    # --- Visual comparison side-by-side ---
    ref_img = Image.open(ref_image).resize((400, 300))
    gen_img = Image.open(gen_path).resize((400, 300))
    side = Image.new("RGB", (800, 300))
    side.paste(ref_img, (0, 0))
    side.paste(gen_img, (400, 0))

    display(Markdown(f"### Evaluation Summary: `{Path(ref_image).name}`"))
    display(
        Markdown(
            f"**Visual:** SSIM `{ssim_score:.3f}`\n"
            f"**Code:** BLEU `{bleu:.3f}` · Edit `{edit_sim:.3f}` · API `{api_sim:.3f}`\n"
            f"**API Details:** Matched `{matched}` · Missed `{missed}` · Extra `{extras}`\n"
            f"**Aggregate:** CodeScore `{code_score:.3f}` · Overall `{overall:.3f}`"
        )
    )
    display(side)

    return {
        "image": ref_image,
        "ssim": ssim_score,
        "bleu": bleu,
        "edit_sim": edit_sim,
        "api_sim": api_sim,
        "code_score": code_score,
        "overall": overall,
        "execution_success": True,
        "instruction": instruction,
        "generated_vega_lite_spec": generated_spec,
        "generated_vega_lite_json": (
            canonical_json_dumps(generated_spec) if generated_spec else None
        ),
        "generated_code": code,
        "ir_exact_match": ir_scores["exact_match"],
        "ir_field_precision": ir_scores["field_precision"],
        "ir_field_recall": ir_scores["field_recall"],
        "ir_field_f1": ir_scores["field_f1"],
        "ir_tree_similarity": ir_scores["tree_similarity"],
    }


def check_evals_batch(model, processor, data, desc, batch_size=4):
    """
    Evaluate the model on a dataset using the same task split as training.

    Args:
        model: Vision-language model
        processor: Model processor/tokenizer
        data: List of dicts with image, code, and optional nonempty vega_spec
        desc: Output directory path / description label
        batch_size: Number of images to generate for simultaneously

    Returns:
        List of evaluation result dicts
    """
    final_results = []
    eval_data = [example for example in data if example.get("image") and example.get("code")]
    skipped = len(data) - len(eval_data)
    if skipped:
        print(f"Skipping {skipped} examples without image/code fields.")

    for i in tqdm(range(0, len(eval_data), batch_size), desc=f"{desc} (batched)"):
        batch = eval_data[i : i + batch_size]
        img_paths = [example["image"] for example in batch]
        instructions = [sample_instruction(example) for example in batch]

        # Generate with the same task split as training:
        # nonempty vega_spec -> Vega-Lite JSON, empty vega_spec -> Python/Matplotlib code.
        batch_codes = generate_code_batch_with_fallback(
            model,
            processor,
            img_paths,
            temperature=0.0,
            samples=batch,
        )

        for example, result_text, instruction in zip(batch, batch_codes, instructions):
            result = evaluate_single_example(
                example,
                result_text,
                op_path=desc,
                instruction=instruction,
            )
            if result:
                final_results.append(result)
            cleanup_memory(clear_gpu=True)

    return final_results


def run_model(model, processor, data, desc="zero_shot", train=False):
    """
    Run evaluation on the full dataset and print/save summary metrics.

    Args:
        model: Vision-language model
        processor: Model processor/tokenizer
        data: List of evaluation examples
        desc: Label for this run (used in output paths and logs)
        train: If True, evaluate on train set; otherwise test set

    Returns:
        (results, res_df, res_successful): raw list, full DataFrame, successful-only DataFrame
    """
    data_source = "train" if train else "test"
    print(f"Evaluating {desc.split('/')[-1]} model on {data_source} set...")
    print("=" * 80)

    path = f"./{desc}"
    os.makedirs(path, exist_ok=True)
    results = check_evals_batch(model, processor, data, path, batch_size=6)

    if len(results) == 0:
        print("No successful evaluations to report.")
        return results, pd.DataFrame(), pd.DataFrame()

    res_df = pd.DataFrame(results)
    res_successful = res_df[res_df["execution_success"] == True]

    print("\n" + "=" * 80)
    print(f"{desc.upper()} MODEL RESULTS ON {data_source.upper()} SET")
    print("=" * 80)
    print(f"Total {data_source} examples:  {len(results)}")
    print(
        f"Successful executions:    {len(res_successful)} ({len(res_successful)/len(results)*100:.1f}%)"
    )

    if len(res_successful) > 0:
        print(f"\nMetrics (mean ± std):")
        print(
            f"  SSIM (Visual):   {res_successful['ssim'].mean():.3f} ± {res_successful['ssim'].std():.3f}"
        )
        print(
            f"  BLEU (Text):     {res_successful['bleu'].mean():.3f} ± {res_successful['bleu'].std():.3f}"
        )
        print(
            f"  Edit Similarity: {res_successful['edit_sim'].mean():.3f} ± {res_successful['edit_sim'].std():.3f}"
        )
        print(
            f"  API Overlap:     {res_successful['api_sim'].mean():.3f} ± {res_successful['api_sim'].std():.3f}"
        )
        print(
            f"  Code Score:      {res_successful['code_score'].mean():.3f} ± {res_successful['code_score'].std():.3f}"
        )
        print(
            f"  Overall Score:   {res_successful['overall'].mean():.3f} ± {res_successful['overall'].std():.3f}"
        )

    print("=" * 80)

    res_df_file = f"{path}/res_df.csv"
    res_df.to_csv(res_df_file, index=False)
    print(f"\n{desc} results saved to {res_df_file}")

    res_successful_file = f"{path}/res_successful.csv"
    res_successful.to_csv(res_successful_file, index=False)
    print(f"Successful results saved to {res_successful_file}")

    with open(f"{path}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Full results saved to {path}/results.json")

    return results, res_df, res_successful
