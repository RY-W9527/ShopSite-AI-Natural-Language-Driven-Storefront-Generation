"""
ShopSite AI - Small Business Website Generator
Block-based page builder: users compose pages from reusable blocks,
LLM (Qwen Coder) rewrites individual block HTML on natural language instruction.
"""

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
import zipfile
import os
import base64
import io
import re
import shutil
import uuid
import urllib.parse
from pathlib import Path
import spaces

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch/transformers not installed. LLM features disabled.")

try:
    from diffusers import AutoPipelineForText2Image
    SD_AVAILABLE = True and TORCH_AVAILABLE
except (ImportError, RuntimeError):
    SD_AVAILABLE = False
    print("⚠️  diffusers not available. Poster generation disabled.")

# ============================================================
# CONFIG
# ============================================================
QWEN_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
QWEN_CODER_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
SD_MODEL_ID      = "stabilityai/sd-turbo"

WORK_DIR     = Path("./workspace");  WORK_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR = Path("./templates")

# ============================================================
# GLOBAL STATE
# ============================================================
current_html        = ""
current_menu_data   = {}
current_site_info   = {}
current_template_key = "warm"
page_blocks         = []   # [{"id": str, "type": str, "html": str}, ...]
sd_pipe             = None

# Menu item HTML template — LLM can rewrite this to change structure
# Placeholders: {name}, {price}, {img_tag}
MENU_ITEM_TEMPLATE_DEFAULT = """\
      <div class="menu-item" data-item-name="{name}">
        {img_tag}
        <div class="menu-item-info">
          <div class="menu-item-name">{name}</div>
          <div class="menu-item-price">{price}</div>
        </div>
      </div>"""
current_menu_item_template = MENU_ITEM_TEMPLATE_DEFAULT

# ============================================================
# BLOCK DEFAULTS  (warm-theme HTML that works with CSS variables)
# ============================================================
BLOCK_DEFAULTS = {
    "Hero Banner": """\
  <div class="hero">
    <div class="hero-badge">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>
      Est. 2024
    </div>
    <h1>Shop Name</h1>
    <p class="hero-tagline">Welcome to our shop</p>
  </div>
  <div class="info-pills">
    <div class="pill">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <span>Open Daily</span>
    </div>
    <div class="pill">
      <svg viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
      <span>Visit Us</span>
    </div>
  </div>""",

    "Promo / Event": """\
  <div class="section-title">
    <h2>What's New</h2>
    <div class="line"></div>
  </div>
  <div class="promo-banner">
    <div class="promo-placeholder">
      <span>Coming Soon</span>
      <small>Stay tuned for updates</small>
    </div>
  </div>""",

    "About / Story": """\
  <div class="section-title">
    <h2>Our Story</h2>
    <div class="line"></div>
  </div>
  <div class="about-card">
    <p>Welcome to our shop. We are passionate about quality and great service. Come visit us and experience the difference.</p>
  </div>""",

    "Contact Info": """\
  <div class="section-title">
    <h2>Find Us</h2>
    <div class="line"></div>
  </div>
  <div class="contact-section">
    <a class="contact-item" href="">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
      </div>
      <div class="contact-text"><div class="label">Phone</div><div class="value">—</div></div>
    </a>
    <div class="contact-item">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
      </div>
      <div class="contact-text"><div class="label">Address</div><div class="value">—</div></div>
    </div>
    <div class="contact-item">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      </div>
      <div class="contact-text"><div class="label">Hours</div><div class="value">—</div></div>
    </div>
  </div>
  <div class="spacer-lg"></div>""",

    "Announcement": """\
  <div style="margin:16px 20px;">
    <div style="background:var(--bg-warm,#F5EDE3);border-radius:var(--card-radius,18px);padding:20px 24px;border-left:4px solid var(--primary);">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:700;margin-bottom:8px;">📢 Notice</div>
      <p style="font-size:15px;line-height:1.6;color:var(--text);">Add your announcement here.</p>
    </div>
  </div>""",

    "Menu Preview": """\
  <div class="section-title">
    <h2>Our Menu</h2>
    <div class="line"></div>
  </div>
  <div style="padding:0 24px 16px;text-align:center;">
    <button onclick="switchPage('menu')" style="background:var(--primary);color:var(--bg);border:none;padding:12px 32px;border-radius:100px;font-size:14px;font-weight:600;cursor:pointer;letter-spacing:0.04em;">View Full Menu →</button>
  </div>""",
}

# ============================================================
# BLOCK HELPERS
# ============================================================
def _uid():
    return str(uuid.uuid4())[:6]

def _block_label(block):
    return f"{block['type']}  [{block['id']}]"

def _find_block(label):
    for b in page_blocks:
        if _block_label(b) == label:
            return b
    return None

def get_block_choices():
    return [_block_label(b) for b in page_blocks]

# ============================================================
# HuggingFace LOCAL INFERENCE
# ============================================================
_hf_models = {}

def load_hf_model(model_id):
    if model_id not in _hf_models:
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _hf_models[model_id] = (tokenizer, model)
        print(f"✅ {model_id} loaded.")
    return _hf_models[model_id]

def _ollama_chat(model, system_prompt, user_message, temperature=0.3):
    """Inner implementation — safe to call from within a @spaces.GPU context."""
    if not TORCH_AVAILABLE:
        return "ERROR: torch/transformers not installed."
    try:
        tokenizer, hf_model = load_hf_model(model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(hf_model.device)
        max_new_tokens = 1024 if model == QWEN_CODER_MODEL else 512
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        return f"ERROR: {e}"

@spaces.GPU
def ollama_chat(model, system_prompt, user_message, temperature=0.3):
    return _ollama_chat(model, system_prompt, user_message, temperature)

def parse_json_from_response(text):
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        m2 = re.search(r'\{.*\}', text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group())
            except json.JSONDecodeError:
                pass
    return {}

# ============================================================
# MENU ZIP
# ============================================================
def process_menu_zip(zip_file):
    menu = {}
    if zip_file is None:
        return menu
    extract_dir = WORK_DIR / "menu_images"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(extract_dir)
    for root, dirs, files in os.walk(extract_dir):
        rel = Path(root).relative_to(extract_dir)
        if str(rel).startswith(('__', '.')):
            continue
        for fname in sorted(files):
            if fname.startswith(('.', '__')):
                continue
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            fpath = Path(root) / fname
            parts = fpath.relative_to(extract_dir).parts
            category = parts[-2] if len(parts) >= 2 else "Menu"
            stem = fpath.stem
            last_us = stem.rfind('_')
            if last_us > 0:
                name_part = stem[:last_us].replace('_', ' ').strip()
                try:
                    price = float(stem[last_us + 1:].strip())
                except ValueError:
                    name_part = stem.replace('_', ' ').strip(); price = 0.0
            else:
                name_part = stem.replace('_', ' ').strip(); price = 0.0
            with open(fpath, 'rb') as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            ext  = fpath.suffix.lower()
            mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
            if category not in menu:
                menu[category] = []
            menu[category].append({
                "name": name_part, "price": price,
                "image_base64": f"data:{mime.get(ext,'image/png')};base64,{img_b64}",
            })
    return menu

# ============================================================
# MENU HTML BUILDERS
# ============================================================
def build_category_tabs(menu_data):
    return "\n      ".join(
        f'<div class="cat-tab" data-cat="{cat}">{cat}</div>'
        for cat in menu_data.keys()
    )

def build_menu_html(menu_data):
    html = ""
    for cat, items in menu_data.items():
        html += f'    <div class="menu-category" data-cat="{cat}">\n'
        html += f'      <div class="menu-category-title">{cat}</div>\n'
        for item in items:
            price_str = f"${item['price']:.2f}" if item['price'] > 0 else ""
            img_src   = item.get('image_base64', '')
            img_tag   = (f'<img class="menu-item-img" src="{img_src}" alt="{item["name"]}" loading="lazy">'
                         if img_src else
                         '<div class="menu-item-img" style="background:linear-gradient(135deg,var(--secondary),var(--primary));opacity:0.3;"></div>')
            html += (current_menu_item_template
                     .replace("{name}", item["name"])
                     .replace("{price}", price_str)
                     .replace("{img_tag}", img_tag)
                     .replace("{img_src}", img_src)
                     .replace("{description}", item.get("description", "")) + "\n")
        html += '    </div>\n'
    return html

# ============================================================
# TEMPLATE ENGINE
# ============================================================
def load_template(key):
    path = TEMPLATE_DIR / f"{key}.html"
    if not path.exists():
        path = TEMPLATE_DIR / "warm.html"
    return path.read_text(encoding='utf-8')

def rebuild_html():
    global current_html
    template  = load_template(current_template_key)
    home_html = "\n".join(b["html"] for b in page_blocks)

    html = template
    html = html.replace("<!-- {{HOME_BLOCKS}} -->", home_html)
    html = html.replace("<!-- {{CATEGORY_TABS}} -->", build_category_tabs(current_menu_data))
    html = html.replace("<!-- {{MENU_ITEMS}} -->",    build_menu_html(current_menu_data))
    html = html.replace("{{SHOP_NAME}}", current_site_info.get("shop_name", "My Shop"))

    # Poster carousel injection (into any .promo-placeholder found in blocks)
    posters = current_site_info.get("posters", [])
    if posters:
        if len(posters) == 1:
            carousel_html = f'<img src="{posters[0]}" alt="Poster" style="width:100%;height:auto;display:block;">'
        else:
            slides = "\n".join(
                f'<div class="ps-slide{"  ps-active" if i==0 else ""}"><img src="{p}" alt="Poster {i+1}" style="width:100%;height:auto;display:block;"></div>'
                for i, p in enumerate(posters)
            )
            dots = "\n".join(
                f'<span class="ps-dot{"  ps-dot-on" if i==0 else ""}" onclick="psGo({i})"></span>'
                for i in range(len(posters))
            )
            carousel_html = (
                f'<div class="ps-wrap">{slides}'
                f'<button class="ps-btn ps-l" onclick="psMove(-1)">&#10094;</button>'
                f'<button class="ps-btn ps-r" onclick="psMove(1)">&#10095;</button>'
                f'<div class="ps-dots">{dots}</div></div>'
            )
        pat  = r'<div class="promo-placeholder"[^>]*>.*?</div>'
        html = re.sub(pat, carousel_html, html, flags=re.DOTALL, count=1)
        carousel_css = (
            ".ps-wrap{position:relative;overflow:hidden;border-radius:var(--card-radius,12px);}"
            ".ps-slide{display:none;}.ps-slide.ps-active{display:block;}"
            ".ps-btn{position:absolute;top:50%;transform:translateY(-50%);background:rgba(0,0,0,0.45);"
            "color:#fff;border:none;padding:10px 14px;font-size:18px;cursor:pointer;z-index:10;border-radius:6px;}"
            ".ps-l{left:8px;}.ps-r{right:8px;}"
            ".ps-dots{position:absolute;bottom:10px;width:100%;text-align:center;}"
            ".ps-dot{display:inline-block;width:8px;height:8px;background:rgba(255,255,255,0.5);"
            "border-radius:50%;margin:0 3px;cursor:pointer;}"
            ".ps-dot.ps-dot-on{background:#fff;}"
        )
        carousel_js = (
            "<script>(function(){var idx=0;"
            "function show(n){var s=document.querySelectorAll('.ps-slide');"
            "var d=document.querySelectorAll('.ps-dot');if(!s.length)return;"
            "idx=(n+s.length)%s.length;"
            "s.forEach(function(e){e.classList.remove('ps-active');});"
            "d.forEach(function(e){e.classList.remove('ps-dot-on');});"
            "s[idx].classList.add('ps-active');"
            "if(d[idx])d[idx].classList.add('ps-dot-on');}"
            "window.psMove=function(d){show(idx+d);};"
            "window.psGo=function(n){show(n);};"
            "})();</script>"
        )
        html = html.replace("</style>", f"\n/* Carousel */\n{carousel_css}\n</style>", 1)
        html = html.replace("</body>", f"\n{carousel_js}\n</body>", 1)

    # Custom CSS overrides
    css = current_site_info.get("custom_css", "")
    if css:
        html = html.replace("</style>", f"\n/* Custom */\n{css}\n</style>", 1)

    current_html = html
    return html

# ============================================================
# BLOCK MANAGEMENT HANDLERS
# ============================================================
def _make_menu_preview_html():
    if not current_menu_data:
        return BLOCK_DEFAULTS["Menu Preview"]
    preview_items = []
    for items in current_menu_data.values():
        for item in items:
            preview_items.append(item)
            if len(preview_items) >= 3:
                break
        if len(preview_items) >= 3:
            break
    rows = ""
    for item in preview_items:
        price_str = f"${item['price']:.2f}" if item.get('price', 0) > 0 else ""
        img_src = item.get('image_base64', '')
        img_html = (f'<img src="{img_src}" alt="{item["name"]}" style="width:56px;height:56px;border-radius:10px;object-fit:cover;flex-shrink:0;">'
                    if img_src else
                    '<div style="width:56px;height:56px;border-radius:10px;background:linear-gradient(135deg,var(--secondary),var(--primary));opacity:0.4;flex-shrink:0;"></div>')
        rows += f"""    <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);">
      {img_html}
      <div>
        <div style="font-size:13px;font-weight:600;color:var(--text);">{item['name']}</div>
        <div style="font-size:13px;color:var(--primary);font-weight:600;">{price_str}</div>
      </div>
    </div>\n"""
    return f"""\
  <div class="section-title">
    <h2>Our Menu</h2>
    <div class="line"></div>
  </div>
  <div style="margin:0 24px;padding:0 20px;background:var(--bg-card);border-radius:var(--card-radius);border:1px solid var(--border);">
{rows}    <div style="padding:14px 0;text-align:center;">
      <button onclick="switchPage('menu')" style="background:var(--primary);color:var(--bg);border:none;padding:10px 28px;border-radius:100px;font-size:13px;font-weight:600;cursor:pointer;">View Full Menu →</button>
    </div>
  </div>"""

def handle_add_block(block_type, selected_label):
    if block_type == "Menu Preview":
        initial_html = _make_menu_preview_html()
    else:
        initial_html = BLOCK_DEFAULTS.get(block_type, "")
    new_block = {"id": _uid(), "type": block_type, "html": initial_html}
    if selected_label and _find_block(selected_label):
        idx = next((i for i, b in enumerate(page_blocks) if _block_label(b) == selected_label), -1)
        page_blocks.insert(idx + 1, new_block)
    else:
        page_blocks.append(new_block)
    rebuild_html()
    choices   = get_block_choices()
    new_label = _block_label(new_block)
    return preview(current_html), gr.update(choices=choices, value=new_label)

def handle_remove_block(selected_label):
    global page_blocks
    if not selected_label:
        return preview(current_html), gr.update()
    page_blocks = [b for b in page_blocks if _block_label(b) != selected_label]
    rebuild_html()
    choices = get_block_choices()
    return preview(current_html), gr.update(choices=choices, value=choices[0] if choices else None)

def handle_move_up(selected_label):
    idx = next((i for i, b in enumerate(page_blocks) if _block_label(b) == selected_label), -1)
    if idx > 0:
        page_blocks[idx - 1], page_blocks[idx] = page_blocks[idx], page_blocks[idx - 1]
        rebuild_html()
    return preview(current_html), gr.update(choices=get_block_choices(), value=selected_label)

def handle_move_down(selected_label):
    idx = next((i for i, b in enumerate(page_blocks) if _block_label(b) == selected_label), -1)
    if 0 <= idx < len(page_blocks) - 1:
        page_blocks[idx], page_blocks[idx + 1] = page_blocks[idx + 1], page_blocks[idx]
        rebuild_html()
    return preview(current_html), gr.update(choices=get_block_choices(), value=selected_label)

# ============================================================
# LLM BLOCK EDITOR
# ============================================================
CODER_SYSTEM = """You are a frontend developer editing a mobile website HTML block.

RULES:
- Output ONLY the modified HTML. No explanation, no markdown fences, no ```html.
- Keep the mobile-friendly layout and existing CSS classes.
- Only change what the instruction asks.
- Do NOT output <html>, <head>, <body>, or <style> tags — only the inner block content.
- You may add inline styles for visual adjustments.
- Preserve existing CSS class names and structure unless explicitly asked to change them."""

def edit_block_with_llm(selected_label, instruction, chat_history):
    chat_history = chat_history or []

    if not current_html:
        chat_history.append({"role": "assistant", "content": "⚠️ Generate a website first in the Create tab."})
        return chat_history, preview(current_html)

    if not selected_label:
        chat_history.append({"role": "assistant", "content": "⚠️ Select a block from the list first."})
        return chat_history, preview(current_html)

    block = _find_block(selected_label)
    if not block:
        chat_history.append({"role": "assistant", "content": "❌ Block not found."})
        return chat_history, preview(current_html)

    user_msg = f"[{block['type']}] {instruction}"
    prompt = f"""Current HTML block:
{block['html']}

Instruction: {instruction}

Output the modified HTML block:"""

    chat_history.append({"role": "user", "content": user_msg})
    raw = ollama_chat(QWEN_CODER_MODEL, CODER_SYSTEM, prompt, temperature=0.3)

    if raw.startswith("ERROR:"):
        chat_history.append({"role": "assistant", "content": f"❌ {raw}"})
        return chat_history, preview(current_html)

    new_html = raw.strip()
    new_html = re.sub(r'^```html?\s*\n?', '', new_html)
    new_html = re.sub(r'\n?```\s*$', '', new_html)

    if '<!DOCTYPE' in new_html or '<html' in new_html.lower():
        chat_history.append({"role": "assistant", "content": "❌ Model returned full page. Try a simpler instruction."})
        return chat_history, preview(current_html)

    block["html"] = new_html
    rebuild_html()
    chat_history.append({"role": "assistant", "content": f"✅ {block['type']} updated!"})
    return chat_history, preview(current_html)

# ============================================================
# CUSTOM BLOCK GENERATOR
# ============================================================
CUSTOM_BLOCK_SYSTEM = """You are generating a new HTML block for a mobile-first restaurant/shop website.
Generate a single self-contained HTML block based on the user's description.

Available CSS variables (already in the page):
  --primary, --secondary, --accent
  --bg, --bg-card, --bg-elevated
  --text, --text-light, --text-muted
  --border, --card-radius, --card-shadow

Available CSS classes:
  .section-title h2   section header with decorative line
  .about-card         padded content card
  .contact-section    list container
  .contact-item       row with icon + text
  .pill               small rounded badge/tag
  .spacer-lg          bottom spacer

RULES:
- Output ONLY the HTML. No explanation, no markdown fences, no ```html.
- Mobile-friendly layout (max ~480px wide).
- Use CSS variables for all colors so it works with light and dark themes.
- Do NOT output <html>, <head>, <body>, or <style> tags."""

def generate_custom_block(description, chat_history):
    chat_history = chat_history or []
    if not current_html:
        chat_history.append({"role": "assistant", "content": "⚠️ Generate a website first."})
        return chat_history, preview(current_html), gr.update()

    user_msg = f"[Custom Block] {description}"
    chat_history.append({"role": "user", "content": user_msg})
    raw = ollama_chat(QWEN_CODER_MODEL, CUSTOM_BLOCK_SYSTEM,
                      f"Generate an HTML block for: {description}", temperature=0.4)

    if raw.startswith("ERROR:"):
        chat_history.append({"role": "assistant", "content": f"❌ {raw}"})
        return chat_history, preview(current_html), gr.update()

    new_html = raw.strip()
    new_html = re.sub(r'^```html?\s*\n?', '', new_html)
    new_html = re.sub(r'\n?```\s*$', '', new_html)

    if '<!DOCTYPE' in new_html or '<html' in new_html.lower():
        chat_history.append({"role": "assistant", "content": "❌ Model returned full page. Try a simpler description."})
        return chat_history, preview(current_html), gr.update()

    new_block = {"id": _uid(), "type": "Custom", "html": new_html}
    page_blocks.append(new_block)
    rebuild_html()
    choices   = get_block_choices()
    new_label = _block_label(new_block)
    chat_history.append({"role": "assistant", "content": "✅ Custom block added! Select it to edit further."})
    return chat_history, preview(current_html), gr.update(choices=choices, value=new_label)

# ============================================================
# STYLE EDITOR  (CSS variable injection, no block needed)
# ============================================================
PARSE_SYSTEM_STYLE = """Parse the user's instruction about visual style into JSON.

Output format: {"actions": [{"prop": "...", "value": "..."}]}

Available props:
- primary, secondary, accent, bg, text (CSS color values like #006400)
- font_heading, font_body (CSS font-family strings)
- card_radius (e.g. "20px")

Examples:
- "Change primary color to forest green" → {"actions":[{"prop":"primary","value":"#228B22"}]}
- "Dark theme, black background" → {"actions":[{"prop":"bg","value":"#1a1a1a"},{"prop":"text","value":"#f0f0f0"}]}
- "Use Georgia for headings" → {"actions":[{"prop":"font_heading","value":"Georgia, serif"}]}

Output ONLY JSON."""

def handle_style_edit(instruction, chat_history):
    chat_history = chat_history or []
    user_msg = f"[Style] {instruction}"
    chat_history.append({"role": "user", "content": user_msg})

    raw = ollama_chat(QWEN_MODEL, PARSE_SYSTEM_STYLE, instruction)
    if raw.startswith("ERROR:"):
        chat_history.append({"role": "assistant", "content": f"❌ {raw}"})
        return chat_history, preview(current_html)

    parsed  = parse_json_from_response(raw)
    actions = parsed.get("actions", [])
    if not actions:
        chat_history.append({"role": "assistant", "content": f"❌ Could not parse. Raw: {raw[:150]}"})
        return chat_history, preview(current_html)

    css_var_map = {
        "primary": "--primary", "secondary": "--secondary",
        "accent": "--accent",   "bg": "--bg", "text": "--text",
        "card_radius": "--card-radius",
    }
    lines, messages = [], []
    for a in actions:
        prop, value = a.get("prop", ""), a.get("value", "")
        if prop == "font_heading":
            lines.append(f".hero h1, .section-title h2, .menu-header h1, .menu-category-title {{ font-family: {value} !important; }}")
            messages.append(f"Changed font_heading")
        elif prop == "font_body":
            lines.append(f"body {{ font-family: {value} !important; }}")
            messages.append(f"Changed font_body")
        elif prop in css_var_map:
            lines.append(f":root {{ {css_var_map[prop]}: {value}; }}")
            messages.append(f"Changed {prop} → {value}")
        else:
            messages.append(f"Unknown prop: {prop}")

    if lines:
        current_site_info["custom_css"] = current_site_info.get("custom_css", "") + "\n" + "\n".join(lines)
    rebuild_html()
    chat_history.append({"role": "assistant", "content": "🎨 " + " · ".join(messages) + "\n\n✅ Done!"})
    return chat_history, preview(current_html)

# ============================================================
# CREATE WEBSITE
# ============================================================
def _make_hero_html(shop_name, desc, hours, addr):
    location_short = addr.split(",")[0][:25] if "," in addr else addr[:25]
    return f"""\
  <div class="hero">
    <div class="hero-badge">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>
      Est. 2024
    </div>
    <h1>{shop_name}</h1>
    <p class="hero-tagline">{desc[:80]}</p>
  </div>
  <div class="info-pills">
    <div class="pill">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <span>{hours or 'Open Daily'}</span>
    </div>
    <div class="pill">
      <svg viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
      <span>{location_short or 'Visit Us'}</span>
    </div>
  </div>"""

def _make_about_html(desc):
    return f"""\
  <div class="section-title">
    <h2>Our Story</h2>
    <div class="line"></div>
  </div>
  <div class="about-card">
    <p>{desc}</p>
  </div>"""

def _make_contact_html(phone, addr, hours):
    addr_enc = urllib.parse.quote_plus(addr)
    return f"""\
  <div class="section-title">
    <h2>Find Us</h2>
    <div class="line"></div>
  </div>
  <div class="contact-section">
    <a class="contact-item" href="tel:{phone}">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
      </div>
      <div class="contact-text"><div class="label">Phone</div><div class="value">{phone or '—'}</div></div>
    </a>
    <a class="contact-item" href="https://maps.google.com/?q={addr_enc}" target="_blank">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
      </div>
      <div class="contact-text"><div class="label">Address</div><div class="value">{addr or '—'}</div></div>
    </a>
    <div class="contact-item">
      <div class="contact-icon">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      </div>
      <div class="contact-text"><div class="label">Hours</div><div class="value">{hours or '—'}</div></div>
    </div>
  </div>
  <div class="spacer-lg"></div>"""

def create_website(shop_name, desc, phone, addr, hours, style, menu_zip, progress=gr.Progress()):
    global page_blocks, current_menu_data, current_site_info, current_template_key, current_menu_item_template
    current_menu_item_template = MENU_ITEM_TEMPLATE_DEFAULT  # reset on each new site

    progress(0.1, desc="Processing menu...")
    current_menu_data = process_menu_zip(menu_zip)
    current_site_info = {
        "shop_name": shop_name or "My Shop",
        "description": desc or "Welcome to our shop!",
        "phone": phone or "", "address": addr or "", "hours": hours or "",
        "custom_css": "", "posters": [], "poster_base64": "",
    }
    current_template_key = {"Warm & Cozy": "warm", "Dark & Elegant": "dark"}.get(style, "warm")

    progress(0.4, desc="Building blocks...")
    page_blocks = [
        {"id": _uid(), "type": "Hero Banner",   "html": _make_hero_html(shop_name or "My Shop", desc or "Welcome!", hours or "", addr or "")},
        {"id": _uid(), "type": "Promo / Event", "html": BLOCK_DEFAULTS["Promo / Event"]},
        {"id": _uid(), "type": "About / Story", "html": _make_about_html(desc or "Welcome to our shop!")},
        {"id": _uid(), "type": "Contact Info",  "html": _make_contact_html(phone or "", addr or "", hours or "")},
    ]

    progress(0.8, desc="Rendering...")
    rebuild_html()

    n_items = sum(len(v) for v in current_menu_data.values())
    n_cats  = len(current_menu_data)
    progress(1.0)
    print(f"[create_website] page_blocks={len(page_blocks)}, html_len={len(current_html)}")
    return preview(current_html), f"✅ {n_items} items · {n_cats} categories"

# ============================================================
# MENU ITEM DATA EDITOR  (add / remove / change price)
# ============================================================
PARSE_SYSTEM_MENU = """Parse the user's instruction about menu items into JSON.

Output format:
{"actions": [{"op": "add|remove|change_price", "name": "...", "price": 0.0, "category": "..."}]}

Examples:
- "Add Iced Mocha $6 in Coffee" → {"actions":[{"op":"add","name":"Iced Mocha","price":6.00,"category":"Coffee"}]}
- "Remove Espresso" → {"actions":[{"op":"remove","name":"Espresso"}]}
- "Change Latte price to $5.50" → {"actions":[{"op":"change_price","name":"Latte","price":5.50}]}

Output ONLY JSON."""

def handle_menu_data_edit(instruction, uploaded_image, chat_history):
    chat_history = chat_history or []
    if not current_html:
        chat_history.append({"role": "assistant", "content": "⚠️ Generate a website first."})
        return chat_history, preview(current_html)

    user_msg = f"[Menu Items] {instruction}"
    chat_history.append({"role": "user", "content": user_msg})
    raw = ollama_chat(QWEN_MODEL, PARSE_SYSTEM_MENU, instruction)
    if raw.startswith("ERROR:"):
        chat_history.append({"role": "assistant", "content": f"❌ {raw}"})
        return chat_history, preview(current_html)

    parsed  = parse_json_from_response(raw)
    actions = parsed.get("actions", [])
    if not actions:
        chat_history.append({"role": "assistant", "content": f"❌ Could not parse. Raw: {raw[:150]}"})
        return chat_history, preview(current_html)

    # Handle optional uploaded image
    img_b64 = None
    if uploaded_image:
        try:
            with open(uploaded_image, 'rb') as f:
                img_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        except Exception:
            pass

    messages = []
    for a in actions:
        op   = a.get("op", "")
        name = a.get("name", "")

        if op == "add":
            cat   = a.get("category", "Menu")
            price = a.get("price", 0.0)
            if cat not in current_menu_data:
                current_menu_data[cat] = []
            current_menu_data[cat].append({"name": name, "price": price, "image_base64": img_b64 or ""})
            messages.append(f"Added {name} (${price:.2f}) to {cat}")

        elif op == "remove":
            found = False
            for cat, items in current_menu_data.items():
                for i, item in enumerate(items):
                    if item["name"].lower() == name.lower():
                        items.pop(i); messages.append(f"Removed {name}"); found = True; break
                if found: break
            if not found:
                messages.append(f"'{name}' not found")

        elif op == "change_price":
            price = a.get("price", 0.0)
            found = False
            for cat, items in current_menu_data.items():
                for item in items:
                    if item["name"].lower() == name.lower():
                        item["price"] = price; messages.append(f"{name} → ${price:.2f}"); found = True; break
                if found: break
            if not found:
                messages.append(f"'{name}' not found")

    rebuild_html()
    extra = " (with photo)" if img_b64 else ""
    chat_history.append({"role": "assistant", "content": "🔧 " + " · ".join(messages) + extra + "\n\n✅ Done!"})
    return chat_history, preview(current_html)

# ============================================================
# MENU STRUCTURE EDITOR
# ============================================================
MENU_STRUCTURE_SYSTEM = """You are editing the HTML template for a single menu item card on a mobile website.

Available placeholders:
  {name}        — item name text  [REQUIRED — must keep]
  {price}       — item price text e.g. "$4.50"  [REQUIRED — must keep]
  {img_tag}     — full <img> element with class="menu-item-img" (76×76 px square)
  {img_src}     — raw image URL/base64 string — MUST be used as: <img src="{img_src}" ...>  NEVER put {img_src} as text content inside a div or span
  {description} — per-item text description string (may be empty)  [use ONLY when adding a prose text description field]

IMAGE RULES (CRITICAL):
- {img_src} is a raw URL string. You MUST embed it like this: <img src="{img_src}" style="..." alt="{name}">
- NEVER write just {img_src} alone as element content — it will render as raw text on the page
- When changing image layout/size, use {img_src} with a custom <img> tag and omit {img_tag}

MULTI-COLUMN LAYOUT RULES:
- To make items display in 2 columns, add a <style> tag at the TOP of the template:
  <style>.menu-category { display:grid; grid-template-columns:1fr 1fr; gap:12px; } .menu-category-title { grid-column:1/-1; }</style>
- Then make the item itself a compact vertical block (display:block, not flex row)

GENERAL RULES:
- Output ONLY the modified HTML template. No explanation, no markdown fences, no ```html.
- Keep {name}, {price}, and at least one of {img_tag} or {img_src}
- For UI decorations (stars, spice icons, badges): write them as static HTML/emoji, NOT as a placeholder
- Use CSS variables for colors (--primary, --bg-card, --text, --text-muted, --border)
- Do NOT add <html>, <head>, or <body> tags. A single <style> tag at the top is allowed for layout.
- Use inline styles freely to override any layout constraints"""

def edit_menu_structure(instruction, chat_history):
    global current_menu_item_template
    chat_history = chat_history or []

    if not current_html:
        chat_history.append({"role": "assistant", "content": "⚠️ Generate a website first."})
        return chat_history, preview(current_html)

    user_msg = f"[Menu Structure] {instruction}"
    chat_history.append({"role": "user", "content": user_msg})
    prompt = f"""Current menu item template:
{current_menu_item_template}

Instruction: {instruction}

Output the modified template (keep {{name}}, {{price}}, {{img_tag}} placeholders):"""

    raw = ollama_chat(QWEN_CODER_MODEL, MENU_STRUCTURE_SYSTEM, prompt, temperature=0.3)

    if raw.startswith("ERROR:"):
        chat_history.append({"role": "assistant", "content": f"❌ {raw}"})
        return chat_history, preview(current_html)

    new_template = raw.strip()
    new_template = re.sub(r'^```html?\s*\n?', '', new_template)
    new_template = re.sub(r'\n?```\s*$', '', new_template)

    # Auto-fix: {img_src} as bare text content (not inside any attribute or CSS url())
    # Valid uses: src="{img_src}", url({img_src}), url('{img_src}'), url("{img_src}")
    # Invalid (shows raw base64 on page): <div>{img_src}</div>
    if "{img_src}" in new_template and not re.search(
        r'(?:src\s*=\s*["\']?|url\s*\(\s*["\']?)\{img_src\}', new_template
    ):
        new_template = new_template.replace(
            "{img_src}",
            '<img src="{img_src}" style="width:100%;height:160px;object-fit:cover;border-radius:8px;" alt="{name}">'
        )

    # Validate required placeholders — {img_tag} can be replaced by {img_src}
    missing = [p for p in ["{name}", "{price}"] if p not in new_template]
    if "{img_tag}" not in new_template and "{img_src}" not in new_template:
        missing.append("{img_tag} or {img_src}")
    if missing:
        chat_history.append({"role": "assistant", "content": f"❌ Model dropped placeholders: {missing}. Try again."})
        return chat_history, preview_menu(current_html)

    current_menu_item_template = new_template

    # If {description} was added and items don't have descriptions yet, auto-generate them
    if "{description}" in new_template:
        items_needing_desc = [
            item for items in current_menu_data.values() for item in items
            if not item.get("description")
        ]
        if items_needing_desc:
            chat_history.append({"role": "assistant", "content": "⏳ Generating item descriptions..."})
            all_names = [item["name"] for item in items_needing_desc]
            prompt = f"Generate descriptions for these menu items: {json.dumps(all_names)}"
            raw2 = ollama_chat(QWEN_MODEL, DESCRIBE_SYSTEM, prompt, temperature=0.7)
            parsed2 = parse_json_from_response(raw2)
            desc_map = {d["name"]: d["description"] for d in parsed2.get("items", []) if "name" in d and "description" in d}
            for items in current_menu_data.values():
                for item in items:
                    if item["name"] in desc_map:
                        item["description"] = desc_map[item["name"]]
            count = len(desc_map)
            chat_history.append({"role": "assistant", "content": f"✅ Structure updated + {count} descriptions generated!"})
        else:
            chat_history.append({"role": "assistant", "content": "✅ Menu item structure updated!"})
    else:
        chat_history.append({"role": "assistant", "content": "✅ Menu item structure updated!"})

    rebuild_html()
    return chat_history, preview_menu(current_html)

DESCRIBE_SYSTEM = """Generate short, appetising descriptions for menu items.
Given a list of item names, output ONLY JSON:
{"items": [{"name": "...", "description": "..."}]}
Each description: 1 sentence, ≤12 words, mouth-watering and relevant to the item.
Output ONLY JSON."""


def reset_menu_structure(chat_history):
    global current_menu_item_template
    current_menu_item_template = MENU_ITEM_TEMPLATE_DEFAULT
    rebuild_html()
    chat_history = (chat_history or []) + [{"role": "assistant", "content": "↩️ Menu structure reset to default."}]
    return chat_history, preview_menu(current_html)

# ============================================================
# POSTER
# ============================================================
def load_sd_model():
    global sd_pipe
    if sd_pipe is not None:
        return sd_pipe
    if not SD_AVAILABLE:
        return None
    sd_pipe = AutoPipelineForText2Image.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        sd_pipe = sd_pipe.to("cuda")
        sd_pipe.enable_attention_slicing()
    return sd_pipe

PARSE_SYSTEM_POSTER = """Extract key promotional text from the user's description for a poster.
Output ONLY JSON: {"title": "...", "offer": "...", "detail": "...", "tagline": "..."}
- title: main headline (2-5 words)
- offer: the key offer (e.g. "20% OFF")
- detail: conditions or date (e.g. "Valid Mon-Fri")
- tagline: short catchy phrase
Output ONLY JSON."""

def extract_poster_info(desc):
    raw = _ollama_chat(QWEN_MODEL, PARSE_SYSTEM_POSTER, desc, temperature=0.4)
    if raw.startswith("ERROR:"):
        return {"title": "Special Offer", "offer": "", "detail": "", "tagline": ""}
    parsed = parse_json_from_response(raw)
    return parsed if parsed else {"title": "Special Offer", "offer": "", "detail": "", "tagline": ""}

def gen_sd_prompt(desc, bg_style):
    if bg_style == "Restaurant atmosphere":
        sys = "Generate a Stable Diffusion prompt for a restaurant/cafe poster background. Show warm interior, food, bokeh. Under 40 words, no quality tags."
        fallback = "warm cozy restaurant interior, wooden table, bokeh lights, appetizing food, soft warm lighting"
        suffix = "no text, no words, no people, professional food photography, 4k, bokeh"
    else:
        sys = "Generate a Stable Diffusion prompt that visually matches the promotion. Style: vibrant, professional. Under 40 words, no quality tags."
        fallback = "happy people enjoying food, warm atmosphere, vibrant colors"
        suffix = "no text, no words, high quality, 4k, professional photography"
    raw = _ollama_chat(QWEN_MODEL, sys, desc, temperature=0.5)
    if raw.startswith("ERROR:"):
        raw = fallback
    return raw.strip().strip('"\'') + ", " + suffix

def compose_poster(bg_img, poster_info):
    img = bg_img.copy().convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for i in range(h // 3, h):
        alpha = int(210 * (i - h // 3) / (h * 2 // 3))
        draw_ov.line([(0, i), (w, i)], fill=(0, 0, 0, min(alpha, 210)))
    img  = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    def get_font(size):
        for fp in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def fit_font(text, base_size, min_size=12):
        max_w = int(w * 0.88)
        size  = base_size
        while size >= min_size:
            font = get_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] <= max_w:
                return font
            size = max(min_size, int(size * 0.82))
        return get_font(min_size)

    def wrap_text(text, font, max_w):
        words, lines, cur = text.split(), [], []
        for word in words:
            test = " ".join(cur + [word])
            if draw.textbbox((0, 0), test, font=font)[2] <= max_w:
                cur.append(word)
            else:
                if cur: lines.append(" ".join(cur))
                cur = [word]
        if cur: lines.append(" ".join(cur))
        return lines or [text]

    def draw_centered(text, font, y, color):
        if not text: return 0
        lines  = wrap_text(text, font, int(w * 0.88))
        line_h = draw.textbbox((0, 0), "A", font=font)[3] + 4
        for i, line in enumerate(lines):
            tw = draw.textbbox((0, 0), line, font=font)[2]
            draw.text(((w - tw) // 2, y + i * line_h), line, font=font, fill=color)
        return line_h * len(lines)

    y = h - int(h * 0.06)
    if poster_info.get("detail"):
        f = fit_font(poster_info["detail"], int(h * 0.040))
        draw_centered(poster_info["detail"], f, y - int(h * 0.045), (200, 200, 200, 255))
        y -= int(h * 0.07)
    if poster_info.get("offer"):
        f = fit_font(poster_info["offer"], int(h * 0.11))
        draw_centered(poster_info["offer"], f, y - int(h * 0.11), (255, 215, 50, 255))
        y -= int(h * 0.13)
    if poster_info.get("title"):
        f = fit_font(poster_info["title"], int(h * 0.065))
        draw_centered(poster_info["title"], f, y - int(h * 0.07), (255, 255, 255, 255))
        y -= int(h * 0.09)
    if poster_info.get("tagline"):
        f = fit_font(poster_info["tagline"], int(h * 0.038))
        draw_centered(poster_info["tagline"], f, y - int(h * 0.04), (160, 230, 160, 255))
    return img.convert("RGB")

def _poster_status():
    n = len(current_site_info.get("posters", []))
    return f"🖼️ {n} poster{'s' if n != 1 else ''} in carousel" if n else "*No posters added yet*"

@spaces.GPU
def _generate_poster_gpu(desc, bg_prompt, uploaded_bg, bg_style):
    """GPU-only: LLM + SD inference + compose. Returns (img, img_b64, info) or (None, None, error_msg)."""
    poster_info = extract_poster_info(desc)
    info = f"📝 **Extracted:** {poster_info.get('title','')} · {poster_info.get('offer','')} · {poster_info.get('detail','')}"

    if uploaded_bg is not None:
        try:
            bg_img = Image.open(uploaded_bg).convert("RGB").resize((512, 768), Image.LANCZOS)
            info += "\n\n🖼️ Using uploaded background"
        except Exception as e:
            return None, None, f"❌ Failed to load image: {e}"
    else:
        sd_prompt = bg_prompt.strip() if bg_prompt and bg_prompt.strip() else gen_sd_prompt(desc, bg_style)
        pipe = load_sd_model()
        if not pipe:
            return None, None, "❌ Failed to load SD model."
        with torch.no_grad():
            bg_img = pipe(prompt=sd_prompt, num_inference_steps=4, guidance_scale=0.0, width=512, height=768).images[0]
        info += f"\n\n🎨 SD prompt: {sd_prompt}"

    final_img = compose_poster(bg_img, poster_info)
    buf = io.BytesIO()
    final_img.save(buf, format='PNG')
    img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return final_img, img_b64, info

def poster_flow(desc, bg_prompt, uploaded_bg, add_to_site, bg_style="Restaurant atmosphere"):
    """Non-GPU wrapper: calls GPU function then updates global state in main process."""
    if not SD_AVAILABLE and uploaded_bg is None:
        return None, "❌ SD not available. Upload a background image.", preview(current_html) if current_html else "", _poster_status()

    final_img, img_b64, info = _generate_poster_gpu(desc, bg_prompt, uploaded_bg, bg_style)

    if final_img is None:
        return None, info, preview(current_html) if current_html else "", _poster_status()

    final_img.save(WORK_DIR / "latest_poster.png")

    if add_to_site and current_html:
        if "posters" not in current_site_info:
            current_site_info["posters"] = []
        current_site_info["posters"].append(img_b64)
        rebuild_html()
        n = len(current_site_info["posters"])
        info += f"\n\n✅ Added! {n} poster{'s' if n > 1 else ''} in carousel."

    return final_img, info, preview(current_html) if current_html else "", _poster_status()

def remove_last_poster():
    posters = current_site_info.get("posters", [])
    if not posters:
        return "*No posters to remove*", preview(current_html)
    posters.pop()
    current_site_info["posters"] = posters
    rebuild_html()
    return _poster_status(), preview(current_html)

def clear_all_posters():
    current_site_info["posters"] = []
    current_site_info["poster_base64"] = ""
    rebuild_html()
    return "*No posters added yet*", preview(current_html)

# ============================================================
# HELPERS
# ============================================================
def esc(html):
    return html.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

def preview(html):
    if not html:
        return '<div style="padding:40px;text-align:center;color:#999;">No website yet. Create one first.</div>'
    return f'<div style="max-width:480px;margin:0 auto;height:700px;overflow-y:auto;border:1px solid #ddd;border-radius:12px;"><iframe srcdoc="{esc(html)}" style="width:100%;height:100%;border:none;"></iframe></div>'

def preview_menu(html):
    """Render preview with the Menu page active instead of Home."""
    if not html:
        return preview(html)
    html_m = html.replace(
        '<div class="page active" id="page-home">',
        '<div class="page" id="page-home">'
    ).replace(
        '<div class="page" id="page-menu">',
        '<div class="page active" id="page-menu">'
    )
    return preview(html_m)

def download():
    if not current_html:
        return None
    p = WORK_DIR / "my_website.html"
    p.write_text(current_html, encoding='utf-8')
    return str(p)

# ============================================================
# UI
# ============================================================
def build_app():
    with gr.Blocks(title="ShopSite AI") as app:
        gr.Markdown("# 🏪 ShopSite AI\n*AI-powered mobile website generator for small businesses*")

        # ---- Tab 1: Create ----
        with gr.Tab("🏗️ Create Website"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Business Info")
                    shop_name = gr.Textbox(label="Shop Name",    placeholder="The Cozy Bean")
                    desc      = gr.Textbox(label="Description",  placeholder="A neighborhood coffee shop...", lines=3)
                    phone     = gr.Textbox(label="Phone",        placeholder="+44 20 7946 0958")
                    addr      = gr.Textbox(label="Address",      placeholder="42 High Street, London")
                    hours     = gr.Textbox(label="Hours",        placeholder="Mon-Fri 7am-7pm")
                    style     = gr.Dropdown(label="Template", choices=["Warm & Cozy", "Dark & Elegant"], value="Warm & Cozy")
                    gr.Markdown("### Menu ZIP\n`Category/Name_Price.png`")
                    menu_zip  = gr.File(label="Upload ZIP", file_types=[".zip"])
                    create_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    status    = gr.Markdown("")
                with gr.Column(scale=1):
                    gr.Markdown("### 📱 Preview")
                    create_prev = gr.HTML()

        # ---- Tab 2: Edit ----
        with gr.Tab("📝 Edit Website"):
            with gr.Row():
                with gr.Column(scale=1):

                    gr.Markdown("### Page Blocks")
                    block_radio = gr.Radio(label="Select a block to edit", choices=[], interactive=True, value=None)

                    with gr.Row():
                        move_up_btn = gr.Button("⬆️ Up",     size="sm")
                        move_dn_btn = gr.Button("⬇️ Down",   size="sm")
                        remove_btn  = gr.Button("🗑️ Remove", size="sm")

                    with gr.Row():
                        add_type = gr.Dropdown(
                            choices=list(BLOCK_DEFAULTS.keys()),
                            value="Hero Banner", label="New block type", scale=2
                        )
                        add_btn = gr.Button("➕ Add", size="sm", scale=1)

                    gr.Markdown("---")
                    gr.Markdown("### ✨ Generate Custom Block")
                    gr.Markdown("*Describe a new block — LLM generates the HTML from scratch*")
                    with gr.Row():
                        custom_block_msg = gr.Textbox(placeholder='e.g. "A loyalty card section with 10 stamp slots" · "Opening hours table"', lines=2, scale=3, label="")
                        custom_block_btn = gr.Button("Generate", variant="primary", scale=1)

                    gr.Markdown("---")
                    gr.Markdown("### Edit Selected Block")
                    gr.Markdown("*Describe what to change — the LLM rewrites the block HTML*")
                    chatbot  = gr.Chatbot(height=280)
                    with gr.Row():
                        edit_msg  = gr.Textbox(placeholder='e.g. "Add a description field to each menu item"', lines=2, scale=3, label="")
                        edit_send = gr.Button("Send", variant="primary", scale=1)

                    gr.Markdown("### Style / Colors")
                    with gr.Row():
                        style_msg  = gr.Textbox(placeholder='e.g. "Make primary color forest green"', scale=3, label="")
                        style_send = gr.Button("Apply", scale=1)

                    gr.Markdown("### Menu Items")
                    gr.Markdown("*Add, remove, or change prices of individual items*")
                    with gr.Row():
                        menu_data_msg   = gr.Textbox(placeholder='e.g. "Add Iced Mocha $6 in Coffee" · "Remove Espresso" · "Change Latte to $5.50"', lines=2, scale=3, label="")
                        menu_data_image = gr.Image(label="📷 Photo", type="filepath", scale=1, height=120)
                        menu_data_send  = gr.Button("Send", variant="primary", scale=1)

                    gr.Markdown("### Menu Item Structure")
                    gr.Markdown("*Rewrite the layout of every menu card — add descriptions, ratings, badges, etc.*")
                    with gr.Row():
                        menu_struct_msg  = gr.Textbox(placeholder='e.g. "Add a short description field below the item name"', lines=2, scale=3, label="")
                        menu_struct_send = gr.Button("Send", variant="primary", scale=1)
                    menu_reset_btn = gr.Button("↩️ Reset menu to default", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### 📱 Live Preview")
                    edit_prev = gr.HTML()

            # Custom block generation
            custom_block_btn.click(
                generate_custom_block, [custom_block_msg, chatbot], [chatbot, edit_prev, block_radio]
            ).then(lambda: "", outputs=custom_block_msg)
            custom_block_msg.submit(
                generate_custom_block, [custom_block_msg, chatbot], [chatbot, edit_prev, block_radio]
            ).then(lambda: "", outputs=custom_block_msg)

            # Wire up block management
            add_btn.click(handle_add_block,    [add_type, block_radio],  [edit_prev, block_radio])
            remove_btn.click(handle_remove_block, block_radio,           [edit_prev, block_radio])
            move_up_btn.click(handle_move_up,  block_radio,              [edit_prev, block_radio])
            move_dn_btn.click(handle_move_down, block_radio,             [edit_prev, block_radio])

            # Wire up LLM editing
            edit_send.click(
                edit_block_with_llm, [block_radio, edit_msg, chatbot], [chatbot, edit_prev]
            ).then(lambda: "", outputs=edit_msg)
            edit_msg.submit(
                edit_block_with_llm, [block_radio, edit_msg, chatbot], [chatbot, edit_prev]
            ).then(lambda: "", outputs=edit_msg)

            # Style
            style_send.click(
                handle_style_edit, [style_msg, chatbot], [chatbot, edit_prev]
            ).then(lambda: "", outputs=style_msg)

            # Menu item data editing
            menu_data_send.click(
                handle_menu_data_edit, [menu_data_msg, menu_data_image, chatbot], [chatbot, edit_prev]
            ).then(lambda: ("", None), outputs=[menu_data_msg, menu_data_image])
            menu_data_msg.submit(
                handle_menu_data_edit, [menu_data_msg, menu_data_image, chatbot], [chatbot, edit_prev]
            ).then(lambda: ("", None), outputs=[menu_data_msg, menu_data_image])

            # Menu structure editing
            menu_struct_send.click(
                edit_menu_structure, [menu_struct_msg, chatbot], [chatbot, edit_prev]
            ).then(lambda: "", outputs=menu_struct_msg)
            menu_struct_msg.submit(
                edit_menu_structure, [menu_struct_msg, chatbot], [chatbot, edit_prev]
            ).then(lambda: "", outputs=menu_struct_msg)
            menu_reset_btn.click(reset_menu_structure, chatbot, [chatbot, edit_prev])

        # ---- Tab 3: Poster ----
        with gr.Tab("🎨 Poster"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Event / Promotion")
                    edesc    = gr.Textbox(label="Promotion description", placeholder="Student discount 20% off, valid Mon-Fri with student ID", lines=2)
                    bg_prompt = gr.Textbox(label="Background prompt (optional)", placeholder="Leave empty to auto-generate", lines=2)
                    bg_style  = gr.Radio(choices=["Restaurant atmosphere", "Match the promotion"], value="Restaurant atmosphere", label="Auto-generate style")
                    gr.Markdown("*Or upload your own background photo*")
                    poster_bg = gr.Image(label="Background Image (optional)", type="filepath")
                    add2site  = gr.Checkbox(label="Add to website carousel", value=True)
                    pbtn      = gr.Button("🎨 Generate Poster", variant="primary", size="lg")
                    pinfo     = gr.Markdown("")
                    gr.Markdown("### Manage Carousel")
                    poster_status = gr.Markdown("*No posters added yet*")
                    with gr.Row():
                        remove_poster_btn = gr.Button("🗑️ Remove last", size="sm")
                        clear_poster_btn  = gr.Button("🗑️ Clear all",   size="sm", variant="stop")
                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    pimg  = gr.Image(label="Poster", type="pil")
                    gr.Markdown("### 📱 Preview")
                    pprev = gr.HTML()

            pbtn.click(poster_flow, [edesc, bg_prompt, poster_bg, add2site, bg_style], [pimg, pinfo, pprev, poster_status])
            remove_poster_btn.click(remove_last_poster, outputs=[poster_status, pprev])
            clear_poster_btn.click(clear_all_posters,   outputs=[poster_status, pprev])

        # ---- Tab 4: Download ----
        with gr.Tab("💾 Download"):
            gr.Markdown("### Download as single HTML\nFully self-contained with embedded images.")
            dbtn  = gr.Button("💾 Download", variant="primary")
            dfile = gr.File(label="Website file")
            dbtn.click(download, outputs=dfile)

        # Create: first update preview+status, then refresh block list
        create_btn.click(
            create_website,
            [shop_name, desc, phone, addr, hours, style, menu_zip],
            [create_prev, status]
        ).then(
            lambda: gr.update(choices=get_block_choices(), value=None),
            inputs=None,
            outputs=block_radio
        ).then(
            lambda: preview(current_html),
            inputs=None,
            outputs=edit_prev
        )

    return app

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  ShopSite AI  —  Block-based page builder")
    print("=" * 50)

    if not TEMPLATE_DIR.exists():
        print(f"\n❌ templates/ not found at {TEMPLATE_DIR.resolve()}")
    else:
        print(f"\n✅ Templates: {', '.join(t.stem for t in TEMPLATE_DIR.glob('*.html'))}")

    if TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ torch available. Device: {device}")
        print(f"   General model: {QWEN_MODEL}")
        print(f"   Coder model:   {QWEN_CODER_MODEL}")
        print("   (Models downloaded from HuggingFace on first use)")
    else:
        print("❌ torch/transformers not installed. Run: pip install torch transformers accelerate")

    if SD_AVAILABLE:
        print(f"✅ SD Turbo ready. CUDA: {torch.cuda.is_available()}")
    else:
        print("⚠️  Poster generation disabled (diffusers not available).")

    print("\n🚀 http://127.0.0.1:7860\n")
    in_colab = "google.colab" in str(__import__("sys").modules)
    demo = build_app()
    if in_colab:
        demo.launch(share=True)
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
