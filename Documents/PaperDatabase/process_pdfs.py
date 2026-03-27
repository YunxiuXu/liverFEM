
import fitz  # PyMuPDF
import openai
import json
import os
import re
from typing import List

openai.api_key = "-" # 请替换为您的实际 OpenAI API Key

def get_available_tags(tags_file_path="Assets/tags.md"):
    """
    Reads the tags from the tags.md file and returns them as a list of strings.
    Each tag will be in the format 'category/tag_name'.
    """
    tags = []
    current_category = ""
    try:
        with open(tags_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("## "):
                    current_category = line.split("## ")[1].replace("/", "")
                elif line and not line.startswith("#") and not line.startswith("cand/"):
                    if "→" not in line: # Exclude synonym lines
                        tags.append(f"{current_category}/{line}")
    except FileNotFoundError:
        print(f"Error: tags file not found at {tags_file_path}")
    return tags

def call_chatgpt_api(text, available_tags):
    """
    调用 ChatGPT API，提取标题、摘要、作者姓名和最相关的标签。
    需要设置 OPENAI_API_KEY 环境变量。
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or openai.api_key)
        tags_list_str = "\\n".join(available_tags) # Format for prompt

        prompt = f"""
        Please extract the title, abstract, and author names from the first page text of the following academic paper. If the text contains multiple titles or abstracts, please choose the one that best fits academic paper conventions. If there is no clear abstract section, please summarize the main content of the text as the abstract.
        From the following list of available tags, please select up to 5 tags that are most relevant to the paper's abstract. Only choose tags from the provided list and provide them in the format 'category/tag_name'.
        
        Available tags:
        {tags_list_str}

        Please return the results in JSON format, containing five keys: "title", "abstract", "author", "tags", and "year". The "tags" should be a list of strings, and "year" should be the publication year (integer). The author should be a comma-separated string of all authors found.
        
        Also, provide a Chinese translation of the abstract. The JSON output should contain six keys: "title", "abstract", "author", "abstract_zh", "tags", and "year".

        Paper text:
        {text}
        """
        response = client.chat.completions.create(
            model="gpt-5-nano", # User specified model
            messages=[
                {"role": "system", "content": "You are a professional academic paper information extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed_response = json.loads(content)
        return (
            parsed_response.get("title", "Title not found"),
            parsed_response.get("abstract", "Abstract not found"),
            parsed_response.get("author", "Author not found"),
            parsed_response.get("abstract_zh", "Chinese abstract not found"),
            parsed_response.get("tags", []),
            parsed_response.get("year", "YYYY")
        )
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return "Title not found", "Abstract not found", "Author not found", "Chinese abstract not found", [], "YYYY"

def extract_title_and_abstract(pdf_path, available_tags):
    """
    只用 ChatGPT API 提取标题、摘要、作者姓名和最相关标签。
    """
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0]
        page_text = first_page.get_text("text")
        doc.close()
        # 调用 ChatGPT API
        title, abstract, author, abstract_zh, tags, year = call_chatgpt_api(page_text, available_tags)
        return title, abstract, author, abstract_zh, tags, year
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return "Title not found", "Abstract not found", "Author not found", "Chinese abstract not found", [], "YYYY"

def find_screenshot_page(doc):
    """
    Finds the most representative page to screenshot using a prioritized approach.
    1. Finds the page with the largest image (ignoring the first page).
    2. Falls back to finding the page with "Figure 1" text.
    3. Falls back to the first page.
    """
    
    # --- Priority 1: Find the largest image ---
    max_image_area = 0
    best_page_for_image = -1
    
    # Iterate from the second page to the end to avoid title page logos
    for page_num in range(1, doc.page_count):
        page = doc[page_num]
        image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
        
        for block in image_blocks:
            bbox = block["bbox"]
            img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if img_area > max_image_area:
                max_image_area = img_area
                best_page_for_image = page_num
    
    # Check if a significant image was found
    if best_page_for_image != -1:
        page_area = doc[0].rect.width * doc[0].rect.height
        # Only accept if the largest image is at least 5% of the page area
        if (max_image_area / page_area) > 0.05:
            print(f"  Found largest image on page {best_page_for_image + 1}. Using this page for screenshot.")
            return doc[best_page_for_image]

    # --- Priority 2: Find "Figure 1" ---
    print("  No significant image found. Falling back to 'Figure 1' text search.")
    figure_pattern = r"(?:Figure|Fig)\.?\s*1\.?"
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text("text") # Extract full text of the page

        # Use Python's re.search for robust regex matching
        if re.search(figure_pattern, page_text, re.IGNORECASE): 
            print(f"  Found variation of 'Figure 1' on page {page_num + 1}.")
            return page

    # --- Priority 3: First Page ---
    print("  'Figure 1' or its variations not found. Using first page for screenshot.")
    return doc[0]

def extract_representative_pixmap(doc, dpi=150):
    """
    Extracts the most representative image using a hybrid strategy.

    Strategy:
    1.  **Priority 1: Find "Figure 1"**. Search all pages for a caption matching "Figure 1"
        and extract the associated image. This is the strongest indicator.
    2.  **Priority 2: Find First Large Image**. If no "Figure 1" is found, search for the
        first large image that is not a header/logo. It applies special filtering rules
        to the first page to avoid logos.
    3.  **Fallback**: If nothing is found, return None.
    """
    # --- Priority 1: Find "Figure 1" and its associated image ---
    figure_pattern = re.compile(r"^\s*(figure|fig|fig\.)\s*1\b", re.IGNORECASE)
    for page in doc:
        # Search for the caption text
        text_instances = page.search_for(figure_pattern.pattern, flags=re.IGNORECASE)
        if not text_instances:
            continue

        caption_bbox = text_instances[0]  # Use the first match on the page

        # Find the closest image *above* the caption
        closest_image_info = None
        min_dist = float('inf')
        
        try:
            image_list = page.get_images(full=True)
        except Exception:
            continue # Skip pages that cause errors
            
        for img_info in image_list:
            img_bbox = page.get_image_bbox(img_info)
            # Check if image is above the caption and they overlap horizontally
            vertical_dist = caption_bbox.y0 - img_bbox.y1
            if vertical_dist >= 0 and max(img_bbox.x0, caption_bbox.x0) < min(img_bbox.x1, caption_bbox.x1):
                if vertical_dist < min_dist:
                    min_dist = vertical_dist
                    closest_image_info = img_info

        if closest_image_info:
            print(f"  Found 'Figure 1' on page {page.number + 1}, extracting associated image.")
            # Combine the bboxes to clip both the image and its caption
            img_bbox = page.get_image_bbox(closest_image_info)
            combined_bbox = img_bbox + caption_bbox
            combined_bbox.inflate(5) # Add a small margin
            return page.get_pixmap(dpi=dpi, clip=combined_bbox)

    # --- Priority 2: Find the first large, non-logo image ---
    print("  'Figure 1' not found. Searching for the first large, non-logo image...")
    MIN_WIDTH = 250
    MIN_HEIGHT = 250
    
    for page in doc:
        try:
            image_list = page.get_images(full=True)
        except Exception:
            continue # Skip pages that cause errors

        for img_info in image_list:
            xref = img_info[0]
            if xref == 0: continue
            
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue # Can't extract pixmap, skip

            # Filter by pixel dimensions
            if pix.width < MIN_WIDTH or pix.height < MIN_HEIGHT:
                continue

            # **Special filter for the first page to avoid logos/headers**
            if page.number == 0:
                img_bbox = page.get_image_bbox(img_info)
                # If image is in the top 15% of the page, it's likely a logo
                if img_bbox.y1 < page.rect.height * 0.15:
                    continue 

            print(f"  Found first significant image (>{MIN_WIDTH}x{MIN_HEIGHT}px) on page {page.number + 1}.")
            if pix.colorspace and pix.colorspace.name not in (fitz.csGRAY.name, fitz.csRGB.name):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            return pix

    print("  No suitable representative image found.")
    return None

def confirm_and_edit_metadata(title, abstract, author, abstract_zh, tags, year):
    """
    显示识别结果并允许用户编辑文件名。
    返回确认后的元数据和文件名。
    """
    print("\n" + "="*80)
    print(f"标题: {title}")
    print(f"作者: {author}")
    print(f"年份: {year}")
    print("="*80)
    
    # 生成 BibTeX 风格文件名
    # 提取第一作者姓氏
    author_parts = [a.strip() for a in author.split(',')]
    first_author_lastname = "UnknownAuthor"
    if author_parts and author_parts[0]:
        first_author_lastname = author_parts[0].split()[-1]
    
    # 提取标题第一个词
    title_first_word = "NoTitle"
    if title and title.split():
        title_first_word = title.split()[0]
    
    # 清理文件名
    first_author_lastname_clean = re.sub(r'[\\/:*?"<>|.]', '', first_author_lastname)
    year_str = str(year)
    title_first_word_clean = re.sub(r'[\\/:*?"<>|.]', '', title_first_word)
    
    base_filename = f"{first_author_lastname_clean}{year_str}{title_first_word_clean}"
    
    # 直接让用户编辑文件名
    new_filename = input(f"确认或编辑文件名 [{base_filename}]: ").strip()
    if new_filename:
        base_filename = new_filename
    
    print(f"✅ 文件名: {base_filename}\n")
    return title, abstract, author, abstract_zh, tags, year, base_filename

def process_pdfs_in_directory(directory="importQueue", output_dir="output", assets_dir="Assets", papers_md_dir="PapersMd", fulltext_dir="fulltext", target_pdf_dir="Pdf"):
    """
    Processes all PDF files in a given directory.

    Args:
        directory (str): The directory containing PDF files.
        output_dir (str): The directory to save the output files (for .txt).
        assets_dir (str): The directory to save image files.
        papers_md_dir (str): The directory to save Markdown files.
        fulltext_dir (str): The directory to save full text files.
        target_pdf_dir (str): The directory to move processed PDFs to.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(papers_md_dir):
        os.makedirs(papers_md_dir)
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    if not os.path.exists(fulltext_dir):
        os.makedirs(fulltext_dir)
    # Ensure the target_pdf_dir also exists
    if not os.path.exists(target_pdf_dir):
        os.makedirs(target_pdf_dir)

    available_tags = get_available_tags()

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            original_pdf_path = os.path.join(directory, filename)
            print(f"Processing {original_pdf_path}...")
            
            title, abstract, author, abstract_zh, tags, year = extract_title_and_abstract(original_pdf_path, available_tags)
            
            # 🔍 交互式确认和编辑环节
            title, abstract, author, abstract_zh, tags, year, new_base_filename = confirm_and_edit_metadata(
                title, abstract, author, abstract_zh, tags, year
            )

            # --- 文件路径定义 ---
            new_pdf_path = os.path.join(target_pdf_dir, f"{new_base_filename}.pdf")
            new_img_filename = f"{new_base_filename}.png"
            new_img_path_in_assets = os.path.join(assets_dir, new_img_filename)
            new_md_path = os.path.join(papers_md_dir, f"{new_base_filename}.md")
            new_fulltext_path = os.path.join(fulltext_dir, f"{new_base_filename}.txt")

            # --- Save Text Output (kept for now) ---
            txt_path = os.path.join(output_dir, f"{new_base_filename}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n\n")
                f.write(f"Author: {author}\n\n")
                f.write(f"Abstract: {abstract}\n\n")
                f.write(f"摘要: {abstract_zh}\n")
                if tags:
                    f.write("tags:\n")
                    for tag in tags:
                        f.write(f"  - {tag}\n")
                f.write("\n")

            # --- Save Markdown Output ---
            with open(new_md_path, "w", encoding="utf-8") as f:
                # 1. YAML front matter for tags
                if tags:
                    f.write("---\n")
                    f.write("tags:\n")
                    for tag in tags:
                        f.write(f"  - {tag}\n")
                    f.write("---\n\n")
                
                # 2. Image (Screenshot) - 使用Obsidian wiki link格式以兼容iOS
                img_filename = os.path.basename(new_img_path_in_assets)
                f.write(f"![[{img_filename}|640]]\n\n")  # 80% width ≈ 640px (假设全宽800px)

                # 3. Chinese Abstract
                f.write(f"## 摘要\n{abstract_zh}\n\n")

                # 4. Title
                f.write(f"# {title}\n\n")
                
                # 5. Author
                f.write(f"**作者:** {author}\n\n")

                # 6. English Abstract
                f.write(f"## Abstract\n{abstract}\n\n")
                
                # 7. Add link to the PDF itself
                rel_pdf_path_from_md = os.path.relpath(new_pdf_path, os.path.dirname(new_md_path))
                f.write(f"[阅读原文]({rel_pdf_path_from_md})\n")

            # --- Save Screenshot directly to Assets with new name ---
            pix = None
            try:
                doc = fitz.open(original_pdf_path) # Open original PDF to extract image
                pix = extract_representative_pixmap(doc)
                doc.close()
            except Exception as e:
                print(f"Could not process file for image extraction {original_pdf_path}: {e}")
            
            if pix:
                try:
                    pix.save(new_img_path_in_assets)
                    print(f"  Saved representative image to {new_img_path_in_assets}")
                except Exception as e:
                    print(f"Could not save screenshot for {original_pdf_path}: {e}")
            else:
                # Use a fallback image if no representative image was found.
                fallback_img_src = os.path.join(assets_dir, "Title_not_found.png")
                fallback_img_dest = new_img_path_in_assets
                try:
                    # Check if fallback exists and destination does not, then copy
                    if os.path.exists(fallback_img_src) and not os.path.exists(fallback_img_dest):
                         import shutil
                         shutil.copy(fallback_img_src, fallback_img_dest)
                         print(f"  Used fallback image for {original_pdf_path}")
                except Exception as e:
                    print(f"Could not copy fallback image for {original_pdf_path}: {e}")

            
            # --- Extract and Save Full Text ---
            try:
                doc_fulltext = fitz.open(original_pdf_path)
                full_text_content = ""
                for page_num in range(doc_fulltext.page_count):
                    full_text_content += doc_fulltext[page_num].get_text("text")
                doc_fulltext.close()

                with open(new_fulltext_path, "w", encoding="utf-8") as f:
                    f.write(full_text_content)
                print(f"  Saved full text to {new_fulltext_path}")
            except Exception as e:
                print(f"Could not extract full text for {original_pdf_path}: {e}")

            # Move and rename the processed PDF immediately
            try:
                if original_pdf_path != new_pdf_path and not os.path.exists(new_pdf_path):
                    os.rename(original_pdf_path, new_pdf_path)
                    print(f"  Moved PDF to {new_pdf_path}")
                elif os.path.exists(new_pdf_path):
                    print(f"  Skipped moving {original_pdf_path}: Target file {new_pdf_path} already exists.")
                    # Optionally, you might want to delete the source file if the target already exists
                    # os.remove(original_pdf_path)
                else:
                    print(f"  PDF {original_pdf_path} already has the desired name and location.")
            except Exception as e:
                print(f"  Could not move PDF {original_pdf_path} to {new_pdf_path}: {e}")


    print(f"\n处理完成。请检查 '{output_dir}'、'{papers_md_dir}'、'{assets_dir}' 和 '{fulltext_dir}' 目录以查看结果。")

if __name__ == "__main__":
    # You can change the directory here if needed.
    # For example: process_pdfs_in_directory("/path/to/your/pdfs")
    process_pdfs_in_directory("importQueue") 