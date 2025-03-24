#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI工具类 - 提供与DeepSeek API通信的功能
"""

import concurrent.futures
import json
import multiprocessing
import os
import random
import threading
import time
from functools import wraps
from typing import Dict, Any, List, Optional, Callable

import requests
from rich.console import Console

# 初始化Rich控制台
console = Console()

class AIError(Exception):
    """AI服务相关错误的基类"""
    pass

class RateLimitError(AIError):
    """API调用频率限制错误"""
    pass

class AuthenticationError(AIError):
    """身份验证错误"""
    pass

class ServerError(AIError):
    """服务器错误"""
    pass

class ConnectionError(AIError):
    """连接错误"""
    pass

class RequestError(AIError):
    """请求错误"""
    pass

def retry(max_retries=3, base_delay=1, max_delay=60, backoff_factor=2, exceptions=(Exception,)):
    """
    装饰器: 为函数添加重试功能
    
    参数:
        max_retries (int): 最大重试次数
        base_delay (float): 初始延迟时间(秒)
        max_delay (float): 最大延迟时间(秒)
        backoff_factor (float): 退避因子
        exceptions (tuple): 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # 如果已达到最大重试次数，则抛出最后捕获的异常
                    if attempt >= max_retries:
                        raise last_exception
                    
                    # 计算延迟时间 (指数退避 + 随机抖动)
                    delay = min(base_delay * (backoff_factor ** attempt) + random.uniform(0, 1), max_delay)
                    time.sleep(delay)
                    
                    # 使用rich输出重试信息
                    retry_msg = f"重试 {attempt+1}/{max_retries}，延迟 {delay:.2f}秒，错误: {str(e)}"
                    console.print(f"[yellow]{retry_msg}[/yellow]")
        return wrapper
    return decorator

class DeepSeekAI:
    """DeepSeek AI API 客户端"""
    
    BASE_URL = "https://api.deepseek.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化DeepSeek API客户端
        
        参数:
            api_key (str, optional): DeepSeek API密钥。如果为None，则尝试从环境变量获取
            max_workers (int, optional): 最大线程数，用于并行请求。如果为None，则根据CPU核心数自动选择
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API密钥未提供，请设置DEEPSEEK_API_KEY环境变量或在初始化时提供")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # 线程安全的存储
        self._session_lock = threading.RLock()

    
    def _handle_error(self, response: requests.Response) -> None:
        """
        处理API错误响应
        
        参数:
            response (Response): 请求响应对象
        
        抛出:
            各种AIError子类异常
        """
        if response.status_code == 200:
            return
            
        try:
            error_data = response.json()
        except json.JSONDecodeError:
            error_data = {"error": {"message": response.text}}
        
        error_message = error_data.get("error", {}).get("message", "未知错误")
        
        if response.status_code == 401:
            raise AuthenticationError(f"认证失败: {error_message}")
        elif response.status_code == 429:
            raise RateLimitError(f"超出API调用频率限制: {error_message}")
        elif 500 <= response.status_code < 600:
            raise ServerError(f"服务器错误: {error_message}")
        else:
            raise RequestError(f"请求错误 ({response.status_code}): {error_message}")
    
    @retry(max_retries=3, exceptions=(RateLimitError, ServerError, ConnectionError, json.JSONDecodeError))
    def generate_text(
        self, 
        prompt: str, 
        model: str = "deepseek-chat", 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成文本响应
        
        参数:
            prompt (str): 提示文本
            model (str): 模型名称
            max_tokens (int): 最大生成令牌数
            temperature (float): 温度参数
            top_p (float): Top-p采样参数
            stream (bool): 是否流式返回
            **kwargs: 其他参数
            
        返回:
            Dict[str, Any]: API响应
        """
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=payload)
            self._handle_error(response)
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"连接DeepSeek API时出错: {str(e)}")
    
    @retry(max_retries=3, exceptions=(RateLimitError, ServerError, ConnectionError, json.JSONDecodeError))
    def generate_report_toc(
        self,
        topic: str,
        pages: int,
        extra_instructions: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成报告的目录结构
        
        参数:
            topic (str): 报告主题
            pages (int): 期望页数
            format (str): 输出格式 ("markdown", "text")
            extra_instructions (str, optional): 额外指令
            **kwargs: 传递给generate_text的其他参数
            
        返回:
            str: 生成的目录内容
        """
        # 构建提示
        prompt = f"""请为一份关于"{topic}"的详细报告生成一份清晰专业的目录大纲。

要求:
1. 这将是一份约{pages}页的Word文档（每页约300字）
2. 目录应该包含引言、多个主要章节、子章节、结论
3. 主题是"{topic}"，目录结构应详细且专业
4. 仅生成目录，不要生成报告内容
5. 目录结构应清晰，章节和子章节应有明确的编号
6. 章节数量应合理，以便能在{pages}页的篇幅内充分展开（通常7-12个章节较为合适）
7. 章节标题应该准确反映内容，既不过于宽泛也不过于具体
8. 使用统一的编号格式：第一级 1., 第二级 1.1, 第三级 1.1.1

目录结构应包含：
- 前言或引言（介绍报告主题和结构）
- 3-8个主要章节（根据主题展开不同方面）
- 适当的子章节（细化主要章节内容）
- 结论或总结（总结报告主要观点）

请直接输出目录结构，不要添加任何解释性文字。
"""
        
        if extra_instructions:
            prompt += f"\n额外要求:\n{extra_instructions}"
            
        response = self.generate_text(prompt=prompt, **kwargs)
        
        try:
            # 提取生成的内容
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise AIError(f"处理API响应时出错: {str(e)}")
    
    @retry(max_retries=3, exceptions=(RateLimitError, ServerError, ConnectionError, json.JSONDecodeError))
    def generate_report_section(
        self,
        topic: str,
        section_title: str,
        toc: str,
        previous_sections: Optional[List[str]] = None,
        section_index: Optional[int] = None,
        total_sections: Optional[int] = None,
        pages: Optional[int] = None,
        target_word_count: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成报告的单个章节
        
        参数:
            topic (str): 报告的主题
            section_title (str): 要生成的章节标题
            toc (str): 完整的目录内容
            format (str): 输出格式 ("markdown", "text")
            previous_sections (List[str], optional): 之前已生成的章节
            section_index (int, optional): 当前章节索引
            total_sections (int, optional): 总章节数
            pages (int, optional): 报告目标页数
            target_word_count (int, optional): 目标字数
            **kwargs: 传递给generate_text的其他参数
            
        返回:
            str: 生成的章节内容
        """
        # 计算已生成的内容总字数
        total_generated_words = 0
        if previous_sections and len(previous_sections) > 0:
            for section in previous_sections:
                total_generated_words += len(section.split())
        
        # 计算目标字数（如果未提供）
        if target_word_count is None and pages is not None:
            # 估算每页约300字
            total_target_words = pages * 300
            
            # 如果有总章节数和当前索引，则根据章节比例分配字数
            if total_sections is not None and section_index is not None:
                # 总字数乘以权重系数，章节重要性权重
                if "引言" in section_title or "绪论" in section_title:
                    weight = 0.8  # 引言稍短
                elif "结论" in section_title or "总结" in section_title:
                    weight = 0.9  # 结论稍短
                elif "参考文献" in section_title:
                    weight = 0.3  # 参考文献非常短
                else:
                    # 主要内容章节平均分配剩余字数
                    weight = 1.2
                
                # 基于权重分配字数
                target_word_count = int(total_target_words / total_sections * weight)
            else:
                # 如果没有章节信息，假定每个章节平均字数
                target_word_count = int(total_target_words / 8)  # 假设平均8个章节
        elif target_word_count is None:
            # 默认目标字数
            target_word_count = 700
            
        # 构建上下文信息
        context = ""
        if previous_sections and len(previous_sections) > 0:
            # 只包含最多3个之前的章节作为上下文，避免提示词过长
            recent_sections = previous_sections[-3:] if len(previous_sections) > 3 else previous_sections
            context = "\n\n".join(recent_sections)
            
        # 构建提示
        prompt = f"""请为一份关于"{topic}"的详细报告生成"{section_title}"章节的内容。

报告目录结构:
{toc}

当前需要生成的章节是: {section_title}

章节信息:
- 当前是第 {section_index+1}/{total_sections} 个章节
- 目标字数: 约 {target_word_count} 字
- 已生成总字数: {total_generated_words} 字
- 总体报告目标: {pages} 页Word文档

要求:
1. 内容应该学术且专业
2. 使用markdown格式进行排版
3. 内容应基于最新的研究和事实
4. 只生成当前章节的内容，不要生成其他章节
5. 保持与报告整体结构的一致性
6. 字数控制在目标字数左右，不要过长或过短
7. 如果是引言或前言章节，应简明地介绍主题和报告结构
8. 如果是结论章节，应总结报告的主要观点和发现
9. 内容应考虑报告已生成部分的上下文，保持逻辑连贯性
10. 严格按照目录结构组织内容，不要重复生成已在目录中的子标题
11. 不要在内容中重复章节标题或再次列出子章节结构

格式说明:
- 请直接开始生成章节内容，不要包含任何章节标题（标题将由系统自动生成）
- 如果当前章节包含子章节，按目录结构直接组织内容，无需输出子章节的标题
- 不要在同一章节内使用不同级别的标题来重复相同的内容
- 直接输出内容，不要添加任何解释
"""

        if context:
            prompt += f"""
已生成的部分内容参考（用于保持一致性）:
{context}
"""
            
        response = self.generate_text(prompt=prompt, **kwargs)
        
        try:
            # 提取生成的内容
            content = response["choices"][0]["message"]["content"]

            return content
        except (KeyError, IndexError) as e:
            raise AIError(f"处理API响应时出错: {str(e)}")

    def get_chapter_level(self, section_title: str) -> int:
        """
        获取章节标题的层级
        
        参数:
            section_title (str): 章节标题
            
        返回:
            int: 章节层级 (1-5)，如果无法识别则返回0
        """
        # 导入正则表达式库
        import re
        
        # 清理标题前后的空白字符
        clean_title = section_title.strip()
        
        # 需要按照从最具体到最不具体的顺序进行匹配
        # 注意：先检查更具体的模式(三级标题)，再检查不太具体的模式(二级标题)，最后检查最不具体的模式(一级标题)
        
        # 三级标题: 1.1.1 或 1.1.1. 等
        if re.match(r'^\d+\.\d+\.\d+\.?\s', clean_title) or re.match(r'^\d+\.\d+\.\d+$', clean_title):
            return 3
            
        # 二级标题: 1.1 或 1.1. 等
        if re.match(r'^\d+\.\d+\.?\s', clean_title) or re.match(r'^\d+\.\d+$', clean_title):
            return 2
            
        # 一级标题: 1. 或 1 或 一、 等
        if re.match(r'^\d+\.?\s', clean_title) or re.match(r'^\d+$', clean_title) or \
           re.match(r'^[一二三四五六七八九十]+、', clean_title) or \
           re.match(r'^第[一二三四五六七八九十]+章', clean_title):
            return 1
            
        # 第四级: (1) 或 （1） 等
        if re.match(r'^\(\d+\)', clean_title) or re.match(r'^（\d+）', clean_title):
            return 4
            
        # 第五级: a. 或 a) 等
        if re.match(r'^[a-z]\)', clean_title) or re.match(r'^[a-z]\.\s', clean_title):
            return 5
            
        # 无法识别层级，但可能为章节标题（例如"引言"、"结论"等）
        if any(key_title in clean_title.lower() for key_title in ["引言", "绪论", "概述", "结论", "总结", "附录"]):
            return 1
            
        # 无法识别层级
        return 0

    def parse_toc(self, toc: str) -> List[str]:
        """
        解析目录内容，提取章节标题列表
        
        参数:
            toc (str): 目录内容
            
        返回:
            List[str]: 章节标题列表
        """
        # 获取所有章节标题
        all_sections = []
        lines = toc.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 跳过一些常见的非章节行
            if any(skip in line.lower() for skip in ['目录', 'contents', 'table of']):
                continue
            
            # 添加章节到列表
            all_sections.append(line)
        
        return all_sections

    def generate_full_report(
        self,
        topic: str,
        pages: int,
        sections: List[str],
        toc: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs
    ) -> str:
        """
        生成完整报告（顺序执行版本）
        
        参数:
            topic (str): 报告主题
            pages (int): 目标页数
            sections (List[str]): 章节标题列表
            toc (str): 目录内容
            progress_callback (Optional[Callable]): 进度回调函数
            **kwargs: 其他参数
            
        返回:
            str: 完整的报告内容
        """
        # 章节总数
        total_sections = len(sections)
        
        if total_sections == 0:
            return ""
        
        # 显示章节信息 - 避免与进度条冲突
        if not progress_callback:
            console.print(f"[blue]章节信息: 共 {total_sections} 个章节[/blue]")
            console.print(
                f"[bold]开始生成报告[/bold]: {topic}\n"
                f"目标页数: {pages} 页 | 章节数量: {total_sections} 个",
                style="blue"
            )
            console.print("[bold]开始顺序生成章节...[/bold]")
        
        # 用于存储生成内容和进度
        section_results = {}
        shared_context = []  # 存储之前生成的章节内容
        completed_count = 0
        
        # 顺序生成所有章节
        for index, section_title in enumerate(sections):
            # 通知开始生成 - 当没有进度回调时才输出
            if not progress_callback:
                console.print(f"[cyan]开始生成[/cyan] 章节 [{index+1}/{total_sections}]: '{section_title}'")
            
            try:
                # 生成章节内容
                content = self.generate_report_section(
                    topic=topic,
                    section_title=section_title,
                    toc=toc,
                    previous_sections=shared_context,  # 传递之前生成的所有内容作为上下文
                    section_index=index,
                    total_sections=total_sections,
                    pages=pages,
                    **kwargs
                )
                
                # 保存结果
                section_results[index] = content
                
                # 更新共享上下文（添加新生成的内容）
                shared_context.append(content)
                
                # 更新进度
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_sections, section_title)
                # 当没有进度回调时才输出完成信息
                elif not progress_callback:
                    console.print(f"[green]完成生成[/green] 章节 [{index+1}/{total_sections}]: '{section_title}'")
                
            except Exception as e:
                error_msg = f"生成章节 '{section_title}' 时出错: {str(e)}"
                console.print(f"[bold red]错误[/bold red]: {error_msg}")
                section_results[index] = f"[生成失败: {str(e)}]"
        
        # 显示完成信息 - 避免与进度条冲突
        if not progress_callback:
            console.print(f"[bold green]报告生成完成[/bold green]，共 {total_sections} 个章节", style="green")
        
        # 合并所有章节内容，按照原始顺序，并添加正确格式的标题
        ordered_content = []
        for i in range(total_sections):
            if i in section_results:
                # 获取对应的章节标题
                section_title = sections[i]
                
                # 根据章节标题格式确定标题级别
                level = self.get_chapter_level(section_title)
                
                # 根据层级添加不同级别的Markdown标题
                if level == 1:
                    header = f"# {section_title}"
                elif level == 2:
                    header = f"## {section_title}"
                elif level == 3:
                    header = f"### {section_title}"
                elif level == 4:
                    header = f"#### {section_title}"
                elif level == 5:
                    header = f"##### {section_title}"
                else:
                    # 如果无法识别层级，尝试从标题中提取数字格式来判断
                    # 例如 "1.2.3 标题" 可以提取出 "1.2.3"
                    import re
                    match = re.match(r'^(\d+(\.\d+)*)\.?\s', section_title)
                    if match:
                        # 计算点号的数量来确定层级
                        level_nums = match.group(1).split('.')
                        if len(level_nums) == 1:
                            header = f"# {section_title}"
                        elif len(level_nums) == 2:
                            header = f"## {section_title}"
                        elif len(level_nums) == 3:
                            header = f"### {section_title}"
                        else:
                            header = f"#### {section_title}"
                    else:
                        # 默认使用一级标题
                        header = f"# {section_title}"
                
                # 组合标题和内容
                formatted_content = f"{header}\n\n{section_results[i]}"
                ordered_content.append(formatted_content)
            else:
                ordered_content.append(f"[章节 {i+1} 生成失败]")
        
        # 合并所有内容
        report_content = "\n\n".join(ordered_content)
        
        return report_content

    
    def is_available(self) -> bool:
        """
        检查API服务是否可用
        
        返回:
            bool: 服务是否可用
        """
        try:
            url = f"{self.BASE_URL}/models"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_recommended_workers(self, task_count: Optional[int] = None) -> int:
        """
        获取推荐的线程数量
        
        参数:
            task_count (int, optional): 任务数量，如果提供则会根据任务数量优化线程数
            
        返回:
            int: 推荐的线程数
        """
        # 基础线程数（基于CPU核心数）
        recommended = self.max_workers
        
        # 如果指定了任务数量，调整线程数不超过任务数
        if task_count is not None and task_count > 0:
            recommended = min(recommended, task_count)
            
        return max(1, recommended)  # 至少返回1个线程
