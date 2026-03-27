#!/usr/bin/env python3
"""
自动Canvas生成器 - 基于标签自动生成Obsidian Canvas
"""

# ==================== 配置区域 ====================
TAG_CONFIGS = [
    {
        "tags": ["device/wearable"],
        "filename": "./Dashboards/wearable_devices.canvas"
    },
    {
        "tags": ["topic/tactile-perception", "topic/neuroanatomy", "method/psychophysics"],
        "filename": "./Dashboards/tactile_neuroanatomy.canvas"
    },
    {
        "tags": ["*"],  # 特殊标记，表示包含所有论文
        "filename": "./Dashboards/all_papers.canvas"
    }
]
# ================================================

import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CanvasGenerator:
    def __init__(self, papers_dir: str, canvas_output_dir: str = None):
        self.papers_dir = Path(papers_dir)
        self.canvas_output_dir = Path(canvas_output_dir) if canvas_output_dir else self.papers_dir.parent
        
    def extract_tags_from_file(self, file_path: Path) -> List[str]:
        """从md文件中提取标签"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 寻找YAML frontmatter
            yaml_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if not yaml_match:
                return []
            
            yaml_content = yaml_match.group(1)
            try:
                metadata = yaml.safe_load(yaml_content)
                if metadata and 'tags' in metadata:
                    return metadata['tags'] if isinstance(metadata['tags'], list) else []
            except yaml.YAMLError:
                pass
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return []
    
    def scan_papers_with_tags(self) -> Dict[str, List[str]]:
        """扫描所有论文文件并提取标签"""
        papers_tags = {}
        
        for md_file in self.papers_dir.glob("*.md"):
            tags = self.extract_tags_from_file(md_file)
            if tags:
                papers_tags[md_file.name] = tags
                
        return papers_tags
    
    def filter_papers_by_tags(self, papers_tags: Dict[str, List[str]], 
                            target_tags: List[str], 
                            match_mode: str = "any") -> List[str]:
        """根据标签筛选论文文件"""
        filtered_papers = []
        
        for paper, tags in papers_tags.items():
            tag_set = set(tags)
            target_set = set(target_tags)
            
            if match_mode == "any":
                # 包含任意一个目标标签
                if tag_set & target_set:
                    filtered_papers.append(paper)
            elif match_mode == "all":
                # 包含所有目标标签
                if target_set.issubset(tag_set):
                    filtered_papers.append(paper)
        
        return filtered_papers
    
    def load_existing_canvas(self, canvas_path: Path) -> Dict:
        """加载现有canvas文件"""
        if canvas_path.exists():
            try:
                with open(canvas_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"nodes": [], "edges": []}
    
    def generate_canvas_layout(self, papers: List[str], 
                             tags: Set[str],
                             canvas_path: Path = None,
                             paper_size: Tuple[int, int] = (400, 400),
                             tag_size: Tuple[int, int] = (250, 60),
                             spacing: int = 100) -> Dict:
        """生成canvas布局 - 增量更新模式"""
        # 加载现有canvas
        existing_canvas = self.load_existing_canvas(canvas_path) if canvas_path else {"nodes": [], "edges": []}
        
        # 提取现有论文节点
        existing_papers = {}
        other_nodes = []
        
        for node in existing_canvas.get("nodes", []):
            if node.get("type") == "file" and node.get("id", "").startswith("paper-"):
                # 从文件路径提取论文名
                file_path = node.get("file", "")
                if file_path.startswith("PapersMd/"):
                    paper_name = file_path.replace("PapersMd/", "")
                    existing_papers[paper_name] = node
            else:
                other_nodes.append(node)
        
        # 计算布局参数 - n行3列布局
        cols = 3
        
        # 找出需要添加的新论文
        new_papers = [p for p in papers if p not in existing_papers]
        
        # 如果没有新论文，保持原有布局
        if not new_papers:
            return existing_canvas
        
        # 计算下一个可用位置
        max_y = 0
        existing_positions = set()
        
        for paper_name, node in existing_papers.items():
            if paper_name in papers:  # 只考虑仍然匹配标签的论文
                x, y = node["x"], node["y"]
                existing_positions.add((x, y))
                max_y = max(max_y, y)
        
        # 为新论文分配位置
        nodes = []
        
        # 添加现有论文（保持原位置）
        for paper in papers:
            if paper in existing_papers:
                nodes.append(existing_papers[paper])
        
        # 为新论文计算位置
        if new_papers:
            # 计算已有论文的总数
            existing_count = len([p for p in papers if p in existing_papers])
            
            for i, paper in enumerate(new_papers):
                # 从现有论文总数开始计算新位置
                total_index = existing_count + i
                col = total_index % cols
                row = total_index // cols
                
                x = col * (paper_size[0] + spacing)
                y = row * (paper_size[1] + spacing)
                
                paper_id = f"paper-{paper[:-3]}"
                
                nodes.append({
                    "id": paper_id,
                    "type": "file",
                    "file": f"PapersMd/{paper}",
                    "x": x,
                    "y": y,
                    "width": paper_size[0],
                    "height": paper_size[1]
                })
        
        # 添加其他节点（非论文节点）
        nodes.extend(other_nodes)
        
        return {
            "nodes": nodes,
            "edges": existing_canvas.get("edges", [])
        }
    
    def generate_canvas_for_tags(self, target_tags: List[str], 
                               output_filename: str = None,
                               match_mode: str = "any") -> str:
        """为指定标签生成canvas文件"""
        # 扫描所有论文
        papers_tags = self.scan_papers_with_tags()
        
        # 检查是否为"所有论文"模式
        if target_tags == ["*"]:
            # 获取所有.md文件
            filtered_papers = [f.name for f in self.papers_dir.glob("*.md")]
        else:
            # 筛选包含目标标签的论文
            filtered_papers = self.filter_papers_by_tags(papers_tags, target_tags, match_mode)
        
        if not filtered_papers:
            print(f"No papers found with tags: {target_tags}")
            return None
        
        # 收集所有相关标签
        all_tags = set()
        if target_tags == ["*"]:
            # 所有论文模式：收集所有标签
            for paper, tags in papers_tags.items():
                if paper in filtered_papers:
                    all_tags.update(tags)
        else:
            # 特定标签模式：只收集相关标签
            for paper in filtered_papers:
                if paper in papers_tags:
                    all_tags.update(papers_tags[paper])
        
        # 生成输出文件路径
        if not output_filename:
            tag_str = "_".join([tag.replace("/", "-") for tag in target_tags])
            output_filename = f"canvas_{tag_str}.canvas"
        
        output_path = self.canvas_output_dir / output_filename
        
        # 生成canvas布局（增量更新）
        canvas_data = self.generate_canvas_layout(filtered_papers, all_tags, output_path)
        
        # 保存canvas文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(canvas_data, f, indent='\t', ensure_ascii=False)
        
        print(f"Canvas generated: {output_path}")
        print(f"Found {len(filtered_papers)} papers with tags: {target_tags}")
        print(f"Total tags in canvas: {len(all_tags)}")
        
        return str(output_path)


# 移除了文件监控相关代码


if __name__ == "__main__":
    # 示例用法
    # 获取脚本所在的目录，并构建 PapersMd 的正确路径
    script_dir = Path(__file__).resolve().parent
    papers_dir = script_dir / "PapersMd"
    
    # 创建生成器
    generator = CanvasGenerator(papers_dir)
    
    # 批量生成canvas
    for config in TAG_CONFIGS:
        print(f"\n生成 {config['filename']}...")
        generator.generate_canvas_for_tags(
            target_tags=config["tags"],
            output_filename=config["filename"]
        )
    
