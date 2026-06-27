# FIXED VERSION - train_utils.py (Key Section)
# This replaces the toxic feedback injection in your original code
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from PIL import Image
import json
from vegalite_utils import (
    VEGA_LITE_INSTRUCTION,
    UNIFIED_INSTRUCTION,
    parse_vegalite_spec,
    canonicalize_vegalite_spec,
    max_workers
)


def convert_to_conversation_clean(sample):
    # Determine whether this sample is a Vega-Lite example or a fallback Python-matplotlib example.
    instruction = UNIFIED_INSTRUCTION
    target_text = None

    # Prefer explicit vega_spec field if present (added during dataset preparation)
    explicit_spec = parse_vegalite_spec(sample.get("vega_spec"))
    canonical_spec = None
    if explicit_spec is not None:
        instruction = VEGA_LITE_INSTRUCTION
        try:
            # Keep Vega targets as compact canonical JSON text for the chat collator.
            canonical_spec = canonicalize_vegalite_spec(explicit_spec)
            target_text = json.dumps(
                canonical_spec,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            # Fall back to stringifying the original spec
            target_text = json.dumps(explicit_spec, sort_keys=True, separators=(",", ":"))
    else:
        # Fallback: use the raw Python code and the unified instruction.
        # Do not auto-generate Vega-Lite here; the vega_spec field is the task source of truth.
        instruction = UNIFIED_INSTRUCTION
        target_text = sample.get("code")

    if target_text is None:
        return None

    with Image.open(sample["image"]) as img:
        image = img.convert("RGB").copy()

    # Build message content. Keep the original text payload for backwards
    # compatibility, and include a structured `vega_spec` object when available.
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction},
    ]

    assistant_content = [{"type": "text", "text": target_text}]
    # if canonical_spec is not None:
    #     # include the parsed/canonical spec as a structured payload
    #     assistant_content.append({"type": "vega_spec", "vega_spec": canonical_spec})

    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    # return {"messages": conversation,}
    return {
        "messages": conversation,
        "instruction": instruction,
        "target_text": target_text,
        "vega_spec": sample.get("vega_spec"),
        "image_path": sample.get("image"),
    }

def create_data_new(data):
    dataset = [None] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_to_conversation_clean, sample): index
            for index, sample in enumerate(data)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting to conversation"
        ):
            index = futures[future]
            try:
                result = future.result()
                if result:
                    dataset[index] = result
            except Exception:
                continue

    return [sample for sample in dataset if sample is not None]

def _extract_image_from_messages(messages):
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "image":
                return content.get("image")
    return None


def _extract_user_instruction(messages):
    if not messages:
        return ""
    for content in messages[0].get("content", []):
        if content.get("type") == "text":
            return content.get("text", "")
    return ""


def _is_vega_training_sample(sample, messages):
    if parse_vegalite_spec(sample.get("vega_spec")) is not None:
        return True
    instruction = sample.get("instruction") or _extract_user_instruction(messages)
    return instruction == VEGA_LITE_INSTRUCTION


def filter_dataset_by_length(
    dataset,
    max_code_length=3000,
    verbose=True,
    processor=None,
    max_total_tokens=None,
    skip_vega_code_length=True,
):
    """
    Parallel version of dataset filtering.
    """

    original_count = len(dataset)

    dropped_for_code = 0
    dropped_for_tokens = 0

    # -------------------------
    # Worker function
    # -------------------------
    def process_sample(sample):
        nonlocal dropped_for_code, dropped_for_tokens

        try:
            messages = sample.get("messages", [])
            if len(messages) < 2:
                return None, 0, 0

            assistant_content = messages[1].get("content", [])
            if not isinstance(assistant_content, list) or len(assistant_content) == 0:
                return None, 0, 0

            target_text = assistant_content[0].get("text", "")
            is_vega_sample = _is_vega_training_sample(sample, messages)

            # The character-length limit is meant for legacy Python code.
            # Vega samples train on vega_spec JSON, not the original code field, so
            # only token-length filtering should decide whether a Vega sample fits.
            if (
                max_code_length is not None
                and not (skip_vega_code_length and is_vega_sample)
                and len(target_text) > max_code_length
            ):
                return None, 1, 0

            # Token length filter
            if processor is not None and max_total_tokens is not None:
                try:
                    image = _extract_image_from_messages(messages)

                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    processor_inputs = processor(
                        text=[prompt],
                        images=[image] if image is not None else None,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                    )

                    token_length = int(processor_inputs["input_ids"].shape[-1])

                    if token_length > max_total_tokens:
                        return None, 0, 1

                except Exception:
                    pass

            return sample, 0, 0

        except Exception:
            return None, 0, 0

    # -------------------------
    # Parallel execution
    # -------------------------
    filtered = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in dataset]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering dataset"):
            result, drop_code, drop_token = future.result()

            dropped_for_code += drop_code
            dropped_for_tokens += drop_token

            if result is not None:
                filtered.append(result)

    # -------------------------
    # Logging
    # -------------------------
    if verbose:
        removed = original_count - len(filtered)
        if removed > 0:
            print(
                f"✓ Filtered dataset: Removed {removed} samples outside the length limits"
            )
            print(f"  Original: {original_count} → Filtered: {len(filtered)} samples")
            if dropped_for_code > 0:
                print(f"  Dropped by legacy code length: {dropped_for_code}")
            if dropped_for_tokens > 0:
                print(f"  Dropped by processor token length: {dropped_for_tokens}")

    return filtered
