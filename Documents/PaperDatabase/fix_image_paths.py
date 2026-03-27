#!/usr/bin/env python3
"""
修复Markdown文件中的图片引用，将HTML img标签转换为Obsidian的wiki-link格式
这样可以在Canvas中正常显示图片，特别是在移动设备上
"""

import os
import re
from pathlib import Path

def convert_img_to_wikilink(content):
    """
    将HTML img标签转换为Obsidian wiki-link格式
    例如: <img src="../Assets/xxx.png" width="50%"/> 
    转换为: ![[xxx.png|500]]
    """
    # 匹配 <img src="../Assets/文件名.png" width="数字%"/>
    pattern = r'<img\s+src="\.\.\/Assets\/([^"]+)"\s+width="(\d+)%"\s*\/>'
    
    def replace_match(match):
        filename = match.group(1)
        width_percent = int(match.group(2))
        # 将百分比转换为像素宽度（假设全宽为800px）
        width_pixels = int(800 * width_percent / 100)
        return f'![[{filename}|{width_pixels}]]'
    
    # 执行替换
    new_content = re.sub(pattern, replace_match, content)
    return new_content

def process_markdown_files(papers_dir):
    """处理PapersMd目录下的所有markdown文件"""
    papers_path = Path(papers_dir)
    
    if not papers_path.exists():
        print(f"错误: 目录不存在 {papers_dir}")
        return
    
    # 获取所有.md文件
    md_files = list(papers_path.glob("*.md"))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    
    converted_count = 0
    
    for md_file in md_files:
        try:
            # 读取文件内容
            with open(md_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 转换图片引用
            new_content = convert_img_to_wikilink(original_content)
            
            # 如果有变化，写回文件
            if new_content != original_content:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ 已转换: {md_file.name}")
                converted_count += 1
            else:
                print(f"- 跳过（无需转换）: {md_file.name}")
                
        except Exception as e:
            print(f"✗ 处理失败 {md_file.name}: {e}")
    
    print(f"\n完成! 共转换了 {converted_count} 个文件")

if __name__ == "__main__":
    # 设置PapersMd目录的路径
    script_dir = Path(__file__).parent
    papers_dir = script_dir / "PapersMd"
    
    print("=" * 60)
    print("Obsidian图片路径修复工具")
    print("=" * 60)
    print(f"目标目录: {papers_dir}")
    print()
    
    # 执行转换
    process_markdown_files(papers_dir)
    
    print()
    print("说明:")
    print("- HTML img标签已转换为Obsidian wiki-link格式")
    print("- 新格式: ![[图片名.png|宽度]]")
    print("- 这种格式在Canvas中可以正确显示，包括移动设备")
    print()

