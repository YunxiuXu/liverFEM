#!/usr/bin/env python3
"""
专门生成所有论文Canvas的脚本
"""

from auto_canvas_generator import CanvasGenerator

def generate_all_papers_canvas():
    """生成包含所有论文的canvas"""
    papers_dir = "./PapersMd"
    
    # 创建生成器
    generator = CanvasGenerator(papers_dir)
    
    # 生成所有论文的canvas
    print("正在生成所有论文的Canvas...")
    generator.generate_canvas_for_tags(
        target_tags=["*"],
        output_filename="./Dashboards/all_papers.canvas"
    )
    print("✅ 所有论文Canvas生成完成！")

if __name__ == "__main__":
    generate_all_papers_canvas()