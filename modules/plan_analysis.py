import re
import sys
import json
import argparse
from pathlib import Path
 
# ── lazy imports so the script gives a clean error if deps are missing ──────
 
def _require(package: str, install: str = ""):
    try:
        return __import__(package)
    except ModuleNotFoundError:
        hint = f"  pip install {install or package}"
        sys.exit(f"[ERROR] Missing dependency '{package}'.\n{hint}")
 
 
# ── regex patterns for dimension-like strings ───────────────────────────────
 
_DIM_PATTERNS = [
    # metric with unit: 12mm, 1.5 mm, 12.5cm, 3.0 m
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|dm|m)\b",
    # imperial: 1/2 in, 3.5", 4 ft, 2'6"
    r'\b\d+(?:[./]\d+)?\s*(?:in(?:ch(?:es)?)?|ft|foot|feet|"|\')?\b',
    # diameter / radius: Ø12, R5, ⌀3.5
    r"[ØøRr⌀]\s*\d+(?:[.,]\d+)?",
    # tolerance: ±0.05, +0.1/-0.0
    r"[±]\s*\d+(?:[.,]\d+)?",
    r"[+]\d+(?:[.,]\d+)?/[-]\d+(?:[.,]\d+)?",
    # angle: 45°, 30 deg
    r"\b\d+(?:[.,]\d+)?\s*(?:°|deg(?:rees?)?)\b",
    # thread: M6x1, M10, UNC 1/4-20
    r"\bM\d+(?:x\d+(?:[.,]\d+)?)?\b",
    r"\b(?:UNC|UNF|BSP|NPT)\s*\d+[/-]\d+\b",
    # dimension expression: 50 x 30 x 10, 100×50
    r"\b\d+(?:[.,]\d+)?\s*[x×]\s*\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?)?\b",
]
 
_DIM_RE = re.compile("|".join(_DIM_PATTERNS), re.IGNORECASE)
 
 
def extract_dimensions_from_text(text: str) -> list[str]:
    """Return all dimension-like tokens found in *text*."""
    matches = _DIM_RE.findall(text)
    # flatten (findall with groups returns tuples)
    flat = []
    for m in matches:
        if isinstance(m, tuple):
            flat.extend(x for x in m if x)
        else:
            flat.append(m)
    # deduplicate while preserving order
    seen = set()
    result = []
    for d in flat:
        d = d.strip()
        if d and d.lower() not in seen:
            seen.add(d.lower())
            result.append(d)
    return result
 
 
# ── Florence-2 inference ─────────────────────────────────────────────────────
 
def load_florence2(device: str = "cpu"):
    """Load Florence-2 model and processor (downloaded on first run)."""
    transformers = _require("transformers")
    torch = _require("torch")
 
    from transformers import AutoProcessor, AutoModelForCausalLM
 
    model_id = "microsoft/Florence-2-base"           # ~900 MB; use -large for better accuracy
    print(f"[INFO] Loading {model_id} …  (first run downloads ~900 MB)")
 
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, processor, device
 
 
def run_florence2_ocr(image, model, processor, device: str) -> str:
    """
    Run Florence-2 OCR on a PIL image and return the extracted text.
    Task token <OCR> returns plain text; <OCR_WITH_REGION> also gives bboxes.
    """
    import torch
 
    task = "<OCR>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
 
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
 
    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task=task, image_size=image.size)
    return parsed.get(task, "")
 
 

def _run_task(image, task_token: str, text_input: str,
              model, processor, device: str,
              max_new_tokens: int = 256) -> str:
    """Generic Florence-2 inference helper."""
    import torch

    prompt = task_token 
    print(f"Running Florence-2 with prompt:\n{prompt}\n")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw, task=task_token, image_size=image.size
    )
    value = parsed.get(task_token, "")
    if isinstance(value, dict):
        value = value.get("answer", str(value))
    return str(value).strip()

def _run_vqa(image, question: str, model, processor, device: str) -> str:
    return _run_task(image, "<DETAILED_CAPTION>", question, model, processor, device,
                     max_new_tokens=128)
 

VQA_QUESTIONS: list[tuple[str, str]] = [
    # holes & bores
    ("hole_count",
     "How many holes are visible on this technical drawing?"),
    # ("hole_diameters",
    #  "What are the diameters of the holes shown in the drawing?"),
    # ("hole_pattern",
    #  "Are the holes arranged in a pattern such as circular, linear, or grid?"),
    # ("threaded_holes",
    #  "Are any holes threaded? If yes, what is the thread specification?"),
    # # overall geometry
    # ("part_shape",
    #  "What is the overall shape of the part, for example rectangular plate, cylinder, or bracket?"),
    # ("symmetry",
    #  "Is the part symmetric? If so, describe the axis of symmetry."),
    # ("overall_dimensions",
    #  "What are the overall length, width, and height or thickness of the part?"),
    # # features
    # ("slots_cutouts",
    #  "Are there any slots, grooves, or cutouts? Describe them briefly."),
    # ("chamfers_fillets",
    #  "Are chamfers or fillets indicated on the drawing?"),
    # ("surface_finish",
    #  "What surface finish or roughness values are specified?"),
    # # material & title block
    # ("material",
    #  "What material is specified for this part?"),
    # ("part_number",
    #  "What is the part number or drawing number shown?"),
    # ("scale",
    #  "What drawing scale is indicated, for example 1:1 or 1:2?"),
    # ("views",
    #  "How many views are shown, such as front, top, side, or isometric?"),
    # # tolerances & fits
    # ("tolerances",
    #  "What geometric or dimensional tolerances are specified on the drawing?"),
    # ("fits",
    #  "Are any shaft or hole fits such as H7/p6 indicated?"),
]


def run_vqa_questions( image, model, processor, device: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for key, question in VQA_QUESTIONS:
        results[key] = _run_vqa(image, question, model, processor, device)
    return results

# ── main ─────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Extract product dimensions from a manufacturing PDF using Florence-2."
    )
    parser.add_argument("pdf", help="Path to the manufacturing plan PDF")
    parser.add_argument(
        "--pages", nargs="+", type=int, default=None,
        metavar="N", help="Page numbers to process (1-based). Default: all pages."
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to save results as JSON (e.g. results.json)"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Rendering resolution in DPI (higher = better OCR, slower). Default: 200."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device: 'cpu', 'cuda', or 'mps'. Default: cpu."
    )
    args = parser.parse_args()
 
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"[ERROR] File not found: {pdf_path}")
 
    # ── Step 1: PDF → images ──────────────────────────────────────────────
    print(f"\n[1/3] Converting PDF to images (DPI={args.dpi}) …")
    images, page_nums = pdf_to_images(str(pdf_path), args.pages, dpi=args.dpi)
    print(f"      {len(images)} page(s) to process: {page_nums}")
 
    # ── Step 2: Load Florence-2 ───────────────────────────────────────────
    print("\n[2/3] Loading Florence-2 …")
    model, processor, device = load_florence2(args.device)
 
    # ── Step 3: OCR + dimension extraction ───────────────────────────────
    print("\n[3/3] Running OCR and extracting dimensions …\n")
 
    all_results: list[dict] = []
    all_dimensions: list[str] = []
 
    for page_num, image in zip(page_nums, images):
        print(f"  ── Page {page_num} ──────────────────────────")
 
        raw_text = run_florence2_ocr(image, model, processor, device)
        dimensions = extract_dimensions_from_text(raw_text)
 
        print(f"  Raw OCR text (truncated):\n    {raw_text[:300].replace(chr(10), ' ')} …")
        if dimensions:
            print(f"  Dimensions found ({len(dimensions)}):")
            for d in dimensions:
                print(f"    • {d}")
        else:
            print("  No dimension patterns found on this page.")
 
        all_results.append({
            "page": page_num,
            "raw_text": raw_text,
            "dimensions": dimensions,
        })
        all_dimensions.extend(d for d in dimensions if d not in all_dimensions)
 
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("SUMMARY — All unique dimensions extracted")
    print("═" * 50)
    if all_dimensions:
        for d in all_dimensions:
            print(f"  {d}")
    else:
        print("  (none found — check DPI or try a cleaner scan)")
 
    # ── Optional JSON output ──────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        payload = {
            "source": str(pdf_path),
            "pages_processed": page_nums,
            "all_dimensions": all_dimensions,
            "by_page": all_results,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n[INFO] Results saved to {out_path}")
 
 
if __name__ == "__main__":
    main()