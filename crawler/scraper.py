"""
数据采集模块
使用 Scrapy + Playwright 爬取技术文档
"""
import logging
import time
from typing import List, Dict, Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

try:
    from playwright.sync_api import sync_playwright, Browser, Page
except ImportError:
    sync_playwright = None
    Browser = None
    Page = None

logger = logging.getLogger(__name__)


class DocumentScraper:
    """文档爬虫"""
    
    def __init__(self, delay: float = 5, respect_robots_txt: bool = True, 
                 user_agent: Optional[str] = None, timeout: int = 30, 
                 max_retries: int = 3, use_config: bool = True):
        """
        初始化爬虫
        
        Args:
            delay: 请求间隔（秒）
            respect_robots_txt: 是否遵守 robots.txt
            user_agent: 用户代理字符串
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            use_config: 是否使用配置文件（如果为 False，则使用传入的参数）
        """
        if sync_playwright is None:
            raise ImportError("需要安装 playwright 库: pip install playwright")
        
        if use_config:
            try:
                from config.config_manager import get_config
                config = get_config()
                crawler_config = config.get("crawler", {})
                
                self.delay = crawler_config.get("delay", delay)
                self.respect_robots_txt = crawler_config.get("respect_robots_txt", respect_robots_txt)
                self.user_agent = crawler_config.get("user_agent", user_agent)
                self.timeout = crawler_config.get("timeout", timeout)
                self.max_retries = crawler_config.get("max_retries", max_retries)
            except Exception as e:
                logger.warning(f"无法加载配置文件，使用默认参数: {e}")
                # 如果配置加载失败，使用传入的参数
                self.delay = delay
                self.respect_robots_txt = respect_robots_txt
                self.user_agent = user_agent
                self.timeout = timeout
                self.max_retries = max_retries
        else:
            # 直接使用传入的参数
            self.delay = delay
            self.respect_robots_txt = respect_robots_txt
            self.user_agent = user_agent
            self.timeout = timeout
            self.max_retries = max_retries
        
        # 设置默认 user_agent
        if not self.user_agent:
            self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        logger.info("文档爬虫已初始化")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
    
    def start(self):
        """启动浏览器"""
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            logger.info("浏览器已启动")
    
    def stop(self):
        """关闭浏览器"""
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        logger.info("浏览器已关闭")
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        检查 robots.txt 是否允许爬取
        
        Args:
            url: 目标 URL
            
        Returns:
            是否允许爬取
        """
        if not self.respect_robots_txt:
            return True
        
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")
        
        # 使用缓存
        if base_url not in self.robots_cache:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robots_cache[base_url] = rp
            except Exception as e:
                logger.warning(f"无法读取 robots.txt ({robots_url}): {e}")
                return True  # 如果无法读取，默认允许
        
        rp = self.robots_cache[base_url]
        return rp.can_fetch(self.user_agent, url)
    
    def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        爬取单个页面
        
        Args:
            url: 页面 URL
            
        Returns:
            包含 url, title, content, html 的字典，失败返回 None
            
        Raises:
            ValueError: 如果 robots.txt 明确禁止爬取且 respect_robots_txt=True
        """
        # 检查 robots.txt
        if not self._check_robots_txt(url):
            if self.respect_robots_txt:
                # 如果明确设置了遵守 robots.txt，且被禁止，应该抛出异常
                error_msg = f"robots.txt 明确禁止爬取此 URL: {url}。请设置 respect_robots_txt=False 或遵守网站规则。"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # 如果设置了不遵守 robots.txt，记录警告但继续
                logger.warning(f"robots.txt 禁止爬取，但已设置不遵守: {url}")
                # 继续执行
        
        if not self.browser:
            self.start()
        
        page: Page = self.browser.new_page()
        
        try:
            # 设置用户代理
            page.set_extra_http_headers({"User-Agent": self.user_agent})
            
            # 访问页面
            page.goto(url, timeout=self.timeout * 1000, wait_until="networkidle")
            
            # 等待页面加载
            time.sleep(1)
            
            # 提取内容
            title = page.title()
            
            # 提取文本内容（可以自定义选择器）
            # 这里简化处理，实际应该根据目标网站调整
            content = page.inner_text("body")
            
            # 获取 HTML（可选）
            html = page.content()
            
            result = {
                "url": url,
                "title": title,
                "content": content,
                "html": html
            }
            
            logger.info(f"页面爬取成功: {url}")
            return result
        
        except Exception as e:
            logger.error(f"页面爬取失败 ({url}): {e}")
            return None
        
        finally:
            page.close()
            # 遵守爬取频率限制
            time.sleep(self.delay)
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        批量爬取 URL
        
        Args:
            urls: URL 列表
            
        Returns:
            爬取结果列表
        """
        results = []
        
        for url in urls:
            result = self.scrape_page(url)
            if result:
                results.append(result)
        
        logger.info(f"批量爬取完成: {len(results)}/{len(urls)} 成功")
        return results
    
    def scrape_site(
        self,
        start_url: str,
        max_pages: int = 100,
        allowed_domains: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        爬取整个网站（简化版，实际应该使用 Scrapy）
        
        Args:
            start_url: 起始 URL
            max_pages: 最大爬取页面数
            allowed_domains: 允许的域名列表
            
        Returns:
            爬取结果列表
        """
        # TODO: 实现完整的网站爬取逻辑
        # 这里简化处理，实际应该使用 Scrapy 框架
        logger.warning("网站爬取功能需要完整实现，建议使用 Scrapy 框架")
        return []


def crawl_documents(urls: List[str]) -> List[str]:
    """
    爬取文档并返回文本内容列表
    
    Args:
        urls: URL 列表
        
    Returns:
        文档文本列表
    """
    with DocumentScraper() as scraper:
        results = scraper.scrape_urls(urls)
        return [r["content"] for r in results if r.get("content")]

