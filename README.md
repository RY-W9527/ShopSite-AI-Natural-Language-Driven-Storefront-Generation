# ShopSite AI

ShopSite AI is an AI-powered storefront generator for small businesses. It lets users create and edit a mobile-friendly website through natural language, including homepage generation, block-level content editing, menu management, and promotional poster creation.

## Demo

- Hugging Face Space: https://huggingface.co/spaces/mvp-lab/GenAI_ShopSiteAI

## Features

- Generate a storefront website from business information
- Edit existing webpage sections with natural language
- Add custom content blocks without rebuilding the whole page
- Manage menu items from uploaded image assets
- Rewrite menu layouts with AI-generated HTML
- Generate promotional posters for campaigns and events
- Export a self-contained HTML website for preview and deployment

## Tech Stack

- Python
- Gradio
- Transformers
- Diffusers
- PIL
- HTML / CSS

## Models Used

- **Qwen2.5-7B-Instruct** for instruction parsing and structured editing
- **Qwen2.5-Coder-14B-Instruct** for HTML rewriting and custom block generation
- **SD-Turbo** for poster background generation

## Repository Structure

```text
.
├── templates/
│   ├── dark.html
│   └── warm.html
├── app.py
├── gitattributes.txt
├── README.md
└── requirements.txt
```

## Project Architecture

### 1. Template Layer
The app loads a base website template from the `templates/` folder. Different themes are supported through separate HTML templates such as `dark.html` and `warm.html`.

### 2. Block-Based Editing
The homepage is composed of reusable content blocks. Instead of regenerating the whole page after every instruction, the system rewrites only the selected block, which improves controllability and reduces unintended edits.

### 3. Menu Processing
The system can process uploaded menu image assets, organize them into structured menu data, and generate menu sections automatically.

### 4. Poster Generation
Promotional posters are generated with a diffusion model and then composed into the storefront as standalone banners or carousel content.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ShopSiteAI.git
cd ShopSiteAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Then open the Gradio URL shown in the terminal.

## Usage

### Generate a website
Provide business information such as shop name, description, contact details, opening hours, and theme preference.

### Edit a section
Select a webpage block and give an instruction such as:
- "Change the hero title to Sunrise Cafe"
- "Make this section more modern and concise"
- "Add a Mother's Day promotion banner"

### Upload menu assets
Upload menu or product images to build structured menu content and preview the generated layout.

### Generate posters
Create promotional visuals and insert them into the page as event or campaign content.

## Why This Project Matters

Many small businesses need a website but do not have the technical expertise or budget to repeatedly update one. ShopSite AI explores how LLMs and generative models can support a practical, lower-barrier workflow for website creation and maintenance.

## Limitations

- Large local models can require significant GPU memory
- Generated HTML may still need manual review before production use
- Poster quality depends on available compute and inference environment

## Future Work

- Better validation for LLM-generated HTML
- More template styles for different business types
- Undo/version history for block edits
- Improved structured menu schema support
- Faster inference and lighter deployment options
