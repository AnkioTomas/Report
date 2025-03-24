#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI报告生成器 - 一个基于AI的命令行报告生成工具
"""

import os
import subprocess
import sys
import tempfile
import time

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from ai import DeepSeekAI, AIError

console = Console()

def clear_screen():
    console.clear()

def print_logo():
    """打印程序logo"""
    # 使用pyfiglet生成ASCII艺术字
    logo_text = pyfiglet.figlet_format("AI Report", font="slant")
    
    # 创建一个渐变色文本
    logo = Text(logo_text)
    logo.stylize("bold")
    
    # 添加渐变色
    for i, char in enumerate(logo_text):
        if char != " " and char != "\n":
            gradient_color = f"color({215 + i % 40})"
            logo.stylize(gradient_color, i, i+1)
    
    # 创建面板
    logo_panel = Panel(
        Align.center(logo),
        border_style="bright_blue",
        box=box.ROUNDED,
        title="[bold yellow]AI报告生成器[/bold yellow]",
        title_align="center",
        subtitle="[italic cyan]v1.0.0[/italic cyan]",
        subtitle_align="right"
    )
    
    console.print(logo_panel)

def print_welcome_message():
    """打印欢迎信息"""
    welcome_table = Table(show_header=False, box=box.SIMPLE)
    welcome_table.add_column("信息", style="cyan", justify="center")
    
    welcome_table.add_row("🌟 欢迎使用AI报告生成器 🌟")
    welcome_table.add_row("这是一个强大的基于AI的报告生成工具")
    welcome_table.add_row("可以帮助您快速创建专业的报告")
    welcome_table.add_row("")
    welcome_table.add_row("[italic]开发者: Ankio Tomas[/italic]")
    
    welcome_panel = Panel(
        welcome_table,
        border_style="green",
        box=box.ROUNDED,
        title="[bold green]欢迎[/bold green]",
        title_align="center"
    )
    
    console.print(welcome_panel)

def show_loading_animation(duration=1.5):
    """显示加载动画"""
    with console.status("[bold blue]正在初始化AI引擎...", spinner="dots"):
        time.sleep(duration)

def save_report_to_file(content, topic, prefix=""):
    """
    保存报告到文件
    
    参数:
        content (str): 报告内容
        topic (str): 报告主题
        format (str): 文件格式 ("markdown", "text")
        prefix (str): 文件名前缀
    
    返回:
        str: 保存的文件路径
    """
    # 创建reports目录（如果不存在）
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # 清理文件名（删除不允许的字符）
    safe_topic = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
    safe_topic = safe_topic.replace(" ", "_")
    
    # 生成文件名 - 简化为仅使用主题名称
    extension = ".md"
    
    if prefix:
        filename = f"{prefix}_{safe_topic}{extension}"
    else:
        filename = f"{safe_topic}{extension}"
        
    filepath = os.path.join(reports_dir, filename)
    
    # 写入文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath

def convert_markdown_to_docx(md_file_path, title=""):
    """
    将Markdown文件转换为Word文档
    
    参数:
        md_file_path (str): Markdown文件路径
        title (str): 文档标题
        
    返回:
        str: 生成的Word文档路径，转换失败则返回None
    """
    try:
        # 检查是否安装了pandoc
        try:
            subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            console.print("[bold red]错误: 未安装Pandoc，无法转换为Word格式[/bold red]")
            console.print("[yellow]请安装Pandoc: https://pandoc.org/installing.html[/yellow]")
            return None
        
        # 构建输出文件路径 - 直接替换扩展名
        docx_file_path = md_file_path.replace('.md', '.docx')
        
        # 创建临时文件用于添加标题样式
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
            # 添加标题和样式
            if title:
                temp_file.write(f"% {title}\n\n")
            
            # 添加原始内容
            with open(md_file_path, "r", encoding="utf-8") as md_file:
                temp_file.write(md_file.read())
            
            temp_file_path = temp_file.name
        
        # 使用pandoc转换
        cmd = [
            "pandoc",
            temp_file_path,
            "-f", "markdown",
            "-t", "docx",
            "-o", docx_file_path,
            "--toc",  # 添加目录
            "--reference-doc=reference.docx" if os.path.exists("reference.docx") else ""  # 使用参考文档样式
        ]
        
        # 移除空参数
        cmd = [arg for arg in cmd if arg]
        
        # 执行转换
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 删除临时文件
        os.unlink(temp_file_path)
        
        return docx_file_path
    
    except Exception as e:
        console.print(f"[bold red]转换Word文档时出错: {e}[/bold red]")
        return None

def check_api_key():
    """检查API密钥是否已设置"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[yellow]注意: 未设置DeepSeek API密钥环境变量[/yellow]")
        console.print("[cyan]您可以通过设置DEEPSEEK_API_KEY环境变量或在生成报告时提供API密钥[/cyan]")
        return False
    return True

def main():
    """主函数"""
    clear_screen()
    print_logo()
    show_loading_animation()
    print_welcome_message()
    
    # 检查API密钥
    check_api_key()
    
    # 检查Pandoc是否已安装
    try:
        subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        console.print("[yellow]注意: 未检测到Pandoc，无法转换为Word文档[/yellow]")
        console.print("[cyan]您可以从 https://pandoc.org/installing.html 安装Pandoc来启用Markdown到Word的转换功能[/cyan]")
    
    # 直接进入报告生成流程
    while True:
        clear_screen()
        console.print("[bold green]AI报告生成器已准备就绪！[/bold green]\n")
        
        # 获取报告主题
        console.print("[yellow]请输入报告的主题：[/yellow]")
        topic = Prompt.ask("[bold cyan]主题", default="未指定主题")
        
        # 添加页数的输入选项
        console.print("\n[yellow]请输入报告页数 (1-100)：[/yellow]")
        pages = IntPrompt.ask(
            "[bold cyan]页数", 
            default=5,
            choices=[str(i) for i in range(1, 101)],
            show_choices=False
        )
        
        # 创建模型参数字典，用于传递给AI客户端
        model_params = {}
        
        # 检查API密钥
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            console.print("[bold red]错误: 未设置DeepSeek API密钥[/bold red]")
            console.print("[yellow]请输入您的API密钥：[/yellow]")
            api_key = Prompt.ask("[bold cyan]API密钥", password=True)
            if not api_key:
                console.print("[bold red]API密钥为空，无法生成报告[/bold red]")
                console.print("\n[bold yellow]按Enter键继续...[/bold yellow]", end="")
                input()
                continue
        
        try:
            # 初始化AI客户端
            console.print("[cyan]正在初始化AI引擎...[/cyan]")
            ai_client = DeepSeekAI(api_key=api_key)
            
            # 记录起始时间
            start_time = time.time()
            
            # 第一步：生成目录
            with console.status("[bold blue]🧠 正在生成报告目录...", spinner="dots"):
                toc = ai_client.generate_report_toc(
                    topic=topic,
                    pages=pages,
                )
            
            # 保存目录并显示
            toc_filepath = save_report_to_file(toc, topic, prefix="TOC")
            console.print(f"\n[bold green]✅ 目录已生成并保存至: [/bold green][cyan]{toc_filepath}[/cyan]")
            
            # 显示目录预览
            toc_panel = Panel(
                Text(toc),
                border_style="blue",
                box=box.ROUNDED,
                title="[bold]报告目录[/bold]",
                title_align="center"
            )
            console.print(toc_panel)
            
            # 解析章节
            sections = ai_client.parse_toc(toc)
            console.print(f"[cyan]共发现 {len(sections)} 个章节[/cyan]")

            # 直接开始生成
            console.print("\n[bold green]🚀 开始生成章节内容...[/bold green]")
            
            # 使用顺序模式生成报告
            try:
                # 修改进度条初始化和跟踪方式
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    "[cyan]{task.completed}/{task.total}",
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    # 显示报告生成的总体信息
                    console.print(f"[blue]章节信息: 共 {len(sections)} 个章节[/blue]")
                    console.print(
                        f"[bold]开始生成报告[/bold]: {topic}\n"
                        f"目标页数: {pages} 页 | 章节数量: {len(sections)} 个",
                        style="blue"
                    )
                    
                    # 添加一个任务占位符，设置总数为章节数量
                    task_id = progress.add_task("[cyan]生成报告章节...", total=len(sections))
                    
                    # 修改回调函数以使用已创建的进度条
                    def progress_callback(current, total, section):
                        # 更新进度和描述，使用更简洁的描述格式
                        progress.update(
                            task_id, 
                            completed=current,
                            description=f"[cyan]生成报告章节 {current}/{total}"
                        )
                        
                        # 单独在进度条下方显示当前处理的章节名称
                        if current > 0:
                            progress.console.print(f"[green]✓ 已完成章节 {current}/{total}[/green]: {section}")
                    
                    # 生成报告
                    report_content = ai_client.generate_full_report(
                        topic=topic,
                        pages=pages,
                        sections=sections,
                        toc=toc,
                        progress_callback=progress_callback,
                        **model_params
                    )
            except Exception as e:
                console.print(f"[bold red]生成报告过程中出错: {e}[/bold red]")
                raise
            
            # 计算生成耗时
            end_time = time.time()
            elapsed = end_time - start_time
            
            # 格式化时间显示（分钟:秒）
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_format = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"
            
            # 显示生成时间
            console.print(f"\n[bold green]🕒 报告生成完成！耗时: {time_format}[/bold green]")
            
            # 保存Markdown格式报告
            md_filepath = save_report_to_file(report_content, topic)
            console.print(f"[cyan]Markdown报告已保存至: [bold]{md_filepath}[/bold][/cyan]")
            
            # 转换为Word文档
            console.print("\n[bold blue]📄 正在转换为Word文档...[/bold blue]")
            docx_filepath = convert_markdown_to_docx(md_filepath, title=topic)
            
            if docx_filepath:
                console.print(f"[cyan]Word文档已保存至: [bold]{docx_filepath}[/bold][/cyan]")
            
            # 询问是否预览
            console.print("\n[yellow]是否预览报告内容？[/yellow]")
            preview = Confirm.ask("预览内容", default=True)
            
            if preview:
                # 创建预览面板
                content_preview = report_content
                if len(content_preview) > 2000:  # 限制预览长度
                    content_preview = content_preview[:2000] + "...\n\n[内容过长，仅显示前2000字符]"
                    
                preview_panel = Panel(
                    Text(content_preview),
                    border_style="blue",
                    box=box.ROUNDED,
                    title="[bold]报告预览[/bold]",
                    title_align="center"
                )
                console.print(preview_panel)
            
            # 询问是否生成新报告或退出
            console.print("\n[yellow]是否继续生成新的报告？[/yellow]")
            continue_gen = Confirm.ask("生成新报告", default=True)
            
            if not continue_gen:
                console.print("[green]感谢使用！再见！[/green]")
                sys.exit(0)
        
        except AIError as e:
            console.print(f"\n[bold red]生成报告时出错: {e}[/bold red]")
        except Exception as e:
            console.print(f"\n[bold red]发生意外错误: {e}[/bold red]")
        
        # 询问是否继续
        console.print("\n[bold yellow]按Enter键继续...[/bold yellow]", end="")
        input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]程序已被用户中断[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]发生错误: {e}[/bold red]")
        sys.exit(1)
