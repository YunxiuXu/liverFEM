# 论文管理系统 - 设置指南

## 重要提示：虚拟环境配置

本项目使用 iCloud 云同步，但虚拟环境（`venv/`）**不应该**在 iCloud 中同步。

### 为什么？

- 虚拟环境包含针对特定系统架构的二进制文件（`.so`、`.dylib` 等）
- 在不同电脑（可能运行不同的 macOS/Python 版本）上同步会导致文件损坏
- iCloud 同步时可能改变文件权限和符号链接

### 解决方案

虚拟环境已添加到 `.gitignore`。启动脚本会在**每台电脑的本地**自动创建和维护虚拟环境。

#### macOS 中排除 iCloud 同步

如果 `venv/` 文件夹仍在 iCloud 中：

1. 打开 Finder，找到项目目录
2. 右键点击 `venv/` 文件夹 → 获取信息
3. 在底部找到 iCloud 驱动器部分，**取消选中**"存储到 iCloud 驱动器"
4. 或者完全删除 `venv/` 文件夹，让启动脚本重新创建

## 快速开始

### 第一次使用（每台电脑都需要做一次）

```bash
# 进入项目目录
cd /path/to/paperDatabase

# 运行启动脚本
python3 run.py
# 或者
bash run.command
```

启动脚本会自动：
1. ✅ 创建虚拟环境（如果不存在）
2. ✅ 检查虚拟环境是否完整（如果损坏则重建）
3. ✅ 安装 `requirements.txt` 中的依赖
4. ✅ 显示功能选择菜单

### 依赖项

所有依赖都列在 `requirements.txt` 中：

- **PyMuPDF** - PDF 处理
- **openai** - OpenAI API 调用
- **PyYAML** - YAML 配置解析

## 故障排除

### 虚拟环境仍然损坏？

如果遇到虚拟环境错误，启动脚本会自动检测并删除损坏的环境，然后重新创建。

如果问题仍然存在，手动删除 `venv/` 文件夹：

```bash
rm -rf venv/
python3 run.py  # 重新创建
```

### 依赖安装失败？

确保网络连接正常，然后重试：

```bash
venv/bin/pip install -r requirements.txt
```

## iCloud 同步最佳实践

✅ **应该同步到 iCloud 的文件：**
- `PapersMd/` - Markdown 论文文件
- `fulltext/` - 全文内容
- `Dashboards/` - Canvas 文件
- `*.py` 脚本
- `requirements.txt`
- `.gitignore`
- `importQueue/` - 待处理的 PDF

❌ **不应该同步到 iCloud 的文件：**
- `venv/` - 虚拟环境（自动排除）
- `__pycache__/` - Python 缓存（自动排除）
- `.vscode/` / `.idea/` - IDE 配置
- `.DS_Store` - macOS 系统文件

---

有问题？删除 `venv/` 文件夹，重新运行脚本就行！✨
