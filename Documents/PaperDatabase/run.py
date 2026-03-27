#!/usr/bin/env python3
"""
论文管理系统启动器
支持PDF处理和Canvas生成两种功能
"""

import subprocess
import sys
import os
import shutil

# 获取当前脚本所在目录（项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 虚拟环境配置 - 创建在本地而非iCloud
# 使用项目目录的hash值作为唯一标识，避免不同项目冲突
PROJECT_HASH = abs(hash(SCRIPT_DIR)) % (10 ** 8)
VENV_DIR = os.path.expanduser(f"~/.local/venvs/paperDatabase_{PROJECT_HASH}")
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")

def check_and_create_venv():
    """检查并创建虚拟环境，如果不存在或损坏则创建。"""
    def is_venv_valid():
        """检查虚拟环境是否有效"""
        if not os.path.exists(VENV_DIR):
            return False
        
        # 检查虚拟环境的关键文件
        python_exe = os.path.join(VENV_DIR, "Scripts", "python") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "pip")
        
        return os.path.exists(python_exe) and os.path.exists(pip_exe)
    
    if not is_venv_valid():
        if os.path.exists(VENV_DIR):
            print(f"虚拟环境 '{VENV_DIR}' 已损坏，正在删除并重新创建...")
            try:
                shutil.rmtree(VENV_DIR)
            except Exception as e:
                print(f"删除损坏的虚拟环境失败: {e}")
                return False
        else:
            print(f"虚拟环境不存在，正在创建到本地目录：")
            print(f"  → {VENV_DIR}")
            # 确保父目录存在
            os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
        
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            print(f"✅ 虚拟环境创建成功（本地目录，不会同步到iCloud）")
        except subprocess.CalledProcessError as e:
            print(f"创建虚拟环境失败: {e}")
            return False
    else:
        print(f"✅ 虚拟环境已存在且有效（本地目录）：{VENV_DIR}")
    return True

def install_requirements():
    """在虚拟环境中安装 requirements.txt 中的依赖。"""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"文件 '{REQUIREMENTS_FILE}' 不存在，跳过安装依赖。")
        return True

    print(f"正在安装 '{REQUIREMENTS_FILE}' 中的依赖...")
    # 确定 pip 路径，兼容 Windows 和 macOS/Linux
    pip_executable = os.path.join(VENV_DIR, "Scripts", "pip") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "pip")

    try:
        subprocess.check_call([pip_executable, "install", "-r", REQUIREMENTS_FILE])
        print("依赖安装成功。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装依赖失败: {e}")
        return False
    except FileNotFoundError:
        print(f"错误: 找不到 pip 可执行文件 '{pip_executable}'。请确保虚拟环境已正确创建。")
        return False

def get_python_executable():
    """获取虚拟环境中的Python解释器路径"""
    return os.path.join(VENV_DIR, "Scripts", "python") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")

def show_menu():
    """显示功能选择菜单"""
    print("=" * 60)
    print("           论文管理系统 - 功能选择")
    print("=" * 60)
    print()
    print("请选择要执行的功能：")
    print()
    print("1. PDF处理 - 将PDF文件转换为Markdown格式")
    print("   ├─ 处理importQueue目录下的PDF文件")
    print("   ├─ 生成论文摘要和全文内容")
    print("   └─ 输出到PapersMd、fulltext、output目录")
    print()
    print("2. Canvas生成 - 基于标签或全部论文生成Obsidian Canvas")
    print("   ├─ 支持多个标签批量生成")
    print("   ├─ n行3列布局，增量更新")
    print("   ├─ 配置文件：auto_canvas_generator.py 开头")
    print("   │  修改TAG_CONFIGS列表来添加新的标签配置")
    print("   └─ 也可生成包含全部论文的Canvas")
    print("      输出到 ./Dashboards/all_papers.canvas")
    print()
    print("0. 退出程序")
    print()
    print("=" * 60)

def get_user_choice():
    """获取用户选择"""
    while True:
        try:
            choice = input("请输入功能编号 (0-2): ").strip()
            if choice in ['0', '1', '2']:
                return choice
            else:
                print("❌ 无效选择，请输入0、1或2")
        except KeyboardInterrupt:
            print("\n\n👋 程序已退出")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 程序已退出")
            sys.exit(0)

def run_pdf_processing():
    """运行PDF处理功能"""
    print("\n🔄 启动PDF处理功能...")
    print("=" * 40)
    
    PROCESS_SCRIPT = "process_pdfs.py"

    def run_process_script():
        """在虚拟环境中运行主要的处理脚本。"""
        if not os.path.exists(PROCESS_SCRIPT):
            print(f"错误: 主处理脚本 '{PROCESS_SCRIPT}' 不存在。")
            return False

        print(f"正在运行主处理脚本 '{PROCESS_SCRIPT}'...")
        python_executable = get_python_executable()
        
        try:
            subprocess.check_call([python_executable, PROCESS_SCRIPT])
            print("✅ 主处理脚本运行完成。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 主处理脚本运行失败: {e}")
            return False
        except FileNotFoundError:
            print(f"错误: 找不到 Python 可执行文件 '{python_executable}'。请确保虚拟环境已正确创建。")
            return False

    # 执行PDF处理流程
    if run_process_script():
        print("\n✅ PDF处理完成！")
    else:
        print("\n❌ PDF处理失败，请检查错误信息。")

def run_canvas_generation():
    """运行Canvas生成功能"""
    print("\n🎨 启动Canvas生成功能...")
    print("=" * 40)
    
    python_executable = get_python_executable()
    
    # 运行基于标签的Canvas生成
    TAGS_CANVAS_SCRIPT = "auto_canvas_generator.py"
    if os.path.exists(TAGS_CANVAS_SCRIPT):
        print("🏷️ 生成基于标签的Canvas...")
        try:
            subprocess.check_call([python_executable, TAGS_CANVAS_SCRIPT])
            print("✅ 基于标签的Canvas生成完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 基于标签的Canvas生成失败: {e}")
        except FileNotFoundError:
            print(f"❌ 错误: 找不到Python解释器。")
            return
    else:
        print(f"❌ 错误: 标签Canvas生成脚本 '{TAGS_CANVAS_SCRIPT}' 不存在。")
    
    print()
    
    # 运行所有论文Canvas生成
    ALL_CANVAS_SCRIPT = "all_papers_canvas.py"
    if os.path.exists(ALL_CANVAS_SCRIPT):
        print("📋 生成所有论文Canvas...")
        try:
            subprocess.check_call([python_executable, ALL_CANVAS_SCRIPT])
            print("✅ 所有论文Canvas生成完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 所有论文Canvas生成失败: {e}")
    else:
        print(f"❌ 错误: 所有论文Canvas生成脚本 '{ALL_CANVAS_SCRIPT}' 不存在。")
    
    print("\n✅ Canvas生成功能全部完成！")


def main():
    """主函数"""
    print("🔧 初始化环境...")
    if not (check_and_create_venv() and install_requirements()):
        print("❌ 环境初始化失败，程序退出。")
        return
    print("✅ 环境初始化完成！\n")
    
    while True:
        show_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print("\n👋 感谢使用，再见！")
            break
        elif choice == '1':
            run_pdf_processing()
        elif choice == '2':
            run_canvas_generation()
        
        # 询问是否继续
        print("\n" + "=" * 60)
        continue_choice = input("是否继续使用其他功能？(y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '是']:
            print("\n👋 感谢使用，再见！")
            break
        print()

if __name__ == "__main__":
    main()