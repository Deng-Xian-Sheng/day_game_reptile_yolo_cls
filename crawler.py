# -*- coding: utf-8 -*-
"""爬取每日大赛搜索结果中的详情页链接和正文图片。

用法：
    python crawler.py
    python crawler.py --fresh
    python crawler.py --max-pages 1 --max-articles 3

输出：
    data/results.json       详情页链接、标题、日期、本地图片路径
    data/images/<文章ID>/   解密后的本地图片
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import config
from common import atomic_write_bytes, atomic_write_json, now_iso

logger = logging.getLogger("crawler")


class CrawlError(RuntimeError):
    """爬取失败。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def build_headers(referer: str | None = None) -> dict[str, str]:
    """构造请求头。"""
    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def request_text(
    session: requests.Session,
    url: str,
    *,
    referer: str | None = None,
    retries: int | None = None,
) -> str:
    """GET 一个 HTML 页面，带重试。"""
    retries = retries or config.REQUEST_RETRIES
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(
                url,
                headers=build_headers(referer),
                timeout=config.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            # 站点未声明编码时，requests 常常误判为 ISO-8859-1
            if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
                resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                wait = min(2.0 * attempt, 6.0)
                logger.warning(
                    "请求失败（%s/%s），%.1f 秒后重试：%s", attempt, retries, wait, url
                )
                time.sleep(wait)
    raise CrawlError(f"请求失败：{url}，最后一次错误：{last_error}") from last_error


def request_bytes(
    url: str,
    *,
    referer: str | None = None,
    retries: int | None = None,
) -> bytes:
    """下载二进制内容，带重试和大小限制。"""
    retries = retries or config.REQUEST_RETRIES
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers=build_headers(referer),
                timeout=config.REQUEST_TIMEOUT,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in content_type:
                    raise CrawlError(f"图片地址返回了 HTML：{url}")

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > config.MAX_IMAGE_BYTES:
                        raise CrawlError(
                            f"图片超过 {config.MAX_IMAGE_BYTES} 字节限制：{url}"
                        )
                    chunks.append(chunk)
                if not chunks:
                    raise CrawlError(f"图片内容为空：{url}")
                return b"".join(chunks)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                wait = min(1.5 * attempt, 5.0)
                logger.warning(
                    "图片下载失败（%s/%s），%.1f 秒后重试：%s",
                    attempt,
                    retries,
                    wait,
                    url,
                )
                time.sleep(wait)
    raise CrawlError(f"图片下载失败：{url}，最后一次错误：{last_error}") from last_error


# ---------------------------------------------------------------------------
# 图片解密
# ---------------------------------------------------------------------------


def detect_image_type(data: bytes) -> str | None:
    """根据文件头识别图片格式，返回扩展名；不是图片则返回 None。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    # AVIF / HEIC 等基于 ISO BMFF 的格式
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "avif"
        if brand in (b"heic", b"heix", b"hevc", b"mif1"):
            return "heic"
    return None


def decrypt_image_bytes(data: bytes) -> tuple[bytes, str]:
    """如果数据是本站加密图片则解密，返回 (图片字节, 扩展名)。

    站点前端会在浏览器里先下载加密文件，再用 AES-128-CBC 解密。
    未加密的图片（直接是 JPEG/PNG 等）会原样返回。
    """
    image_type = detect_image_type(data)
    if image_type:
        return data, image_type

    if len(data) % 16 != 0:
        raise CrawlError(f"数据既不是图片，长度也不是 AES 块大小的整数倍：{len(data)}")

    decryptor = Cipher(
        algorithms.AES(config.IMAGE_AES_KEY),
        modes.CBC(config.IMAGE_AES_IV),
    ).decryptor()
    decrypted = decryptor.update(data) + decryptor.finalize()

    # 正常情况下是 PKCS7 填充；个别情况下若去填充失败，直接使用原始解密结果，
    # 图片解码器通常可以忽略末尾多余的填充字节。
    try:
        unpadder = padding.PKCS7(128).unpadder()
        decrypted = unpadder.update(decrypted) + unpadder.finalize()
    except ValueError:
        logger.debug("AES 去填充失败，保留原始解密结果")

    image_type = detect_image_type(decrypted)
    if not image_type:
        raise CrawlError("AES 解密后仍不是可识别的图片格式")
    return decrypted, image_type


# ---------------------------------------------------------------------------
# HTML 解析
# ---------------------------------------------------------------------------


def parse_search_page(html: str, page_url: str) -> tuple[list[dict[str, str]], int | None]:
    """解析搜索结果页，返回 (文章列表, 总页数)。"""
    soup = BeautifulSoup(html, "html.parser")
    articles: list[dict[str, str]] = []
    seen: set[str] = set()

    cards = soup.select("#archive article") or soup.select("article")
    for card in cards:
        link = card.select_one('a[href*="/archives/"]')
        if not link or not link.get("href"):
            continue
        url = urljoin(page_url, link["href"].strip())
        if url in seen:
            continue
        seen.add(url)

        title_node = card.select_one(".post-card-title") or card.select_one("h1, h2")
        title = title_node.get_text(" ", strip=True) if title_node else ""

        date_node = card.select_one('meta[itemprop="datePublished"]')
        published_at = date_node.get("content", "") if date_node else ""

        author_node = card.select_one('meta[itemprop="author"]')
        author = author_node.get("content", "") if author_node else ""

        articles.append(
            {
                "url": url,
                "title": title,
                "published_at": published_at.strip(),
                "author": author.strip(),
            }
        )

    total_pages: int | None = None
    page_info = soup.select_one(".page-info")
    if page_info:
        match = re.search(r"(\d+)\s*/\s*(\d+)", page_info.get_text(" ", strip=True))
        if match:
            total_pages = int(match.group(2))
    if total_pages is None:
        # 兜底：从分页条里找最大页码
        numbers = []
        for node in soup.select(".page-navigator a"):
            text = node.get_text(strip=True)
            if text.isdigit():
                numbers.append(int(text))
        if numbers:
            total_pages = max(numbers)

    return articles, total_pages


def is_useful_image_url(url: str) -> bool:
    """过滤站点的占位图、主题资源和广告图。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path.lower()
    if path.endswith(("zw.png", "banner.png", "avatar.jpg", "favicon.ico")):
        return False
    if "/usr/themes/" in path or "/usr/plugins/" in path:
        return False
    if "/assets/" in path and path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return False
    return True


def parse_detail_page(
    html: str, detail_url: str
) -> tuple[str, str, str, list[str]]:
    """解析详情页，返回 (标题, 发布时间, 作者, 正文图片 URL 列表)。"""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    h1 = soup.select_one("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title:
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title:
            title = (og_title.get("content") or "").strip()

    published_at = ""
    date_node = soup.select_one('meta[itemprop="datePublished"]')
    if date_node:
        published_at = (date_node.get("content") or "").strip()
    if not published_at:
        time_node = soup.select_one("time[datetime]")
        if time_node:
            published_at = (time_node.get("datetime") or "").strip()

    author = ""
    author_node = soup.select_one('meta[itemprop="author"]')
    if author_node:
        author = (author_node.get("content") or "").strip()

    content = soup.select_one(".post-content")
    if content is None:
        # 详情页结构变化时，退化为从整页里找文章区域
        content = soup.select_one('article[itemtype*="BlogPosting"]') or soup

    image_urls: list[str] = []
    seen: set[str] = set()
    for img in content.select("img"):
        # 正文图片的真实地址通常在 data-xkrkllgl 里，src 是占位图
        for attr in (
            "data-xkrkllgl",
            "data-src",
            "data-original",
            "data-lazy-src",
            "src",
        ):
            raw = (img.get(attr) or "").strip()
            if not raw or raw.startswith("data:"):
                continue
            candidate = urljoin(detail_url, raw)
            if not is_useful_image_url(candidate):
                continue
            if candidate not in seen:
                seen.add(candidate)
                image_urls.append(candidate)
            break

    return title, published_at, author, image_urls


# ---------------------------------------------------------------------------
# 文章与图片下载
# ---------------------------------------------------------------------------


def article_slug(url: str) -> str:
    """用文章 ID 作为本地目录名。"""
    match = re.search(r"/archives/(\d+)", url)
    if match:
        return match.group(1)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def alternate_image_urls(image_url: str) -> list[str]:
    """部分老文章的图片地址会多包一层原始 CDN 域名，这里生成候选地址。

    例如：
        https://pic.sugfew.cn/pic.old-cdn.cn/upload_01/x.jpg
    会尝试：
        https://pic.old-cdn.cn/upload_01/x.jpg
        https://pic.sugfew.cn/upload_01/x.jpg
    """
    parsed = urlparse(image_url)
    first, sep, rest = parsed.path.lstrip("/").partition("/")
    if not sep or not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}", first, re.I):
        return []

    candidates = [
        f"{parsed.scheme}://{first}/{rest}",
        f"{parsed.scheme}://{parsed.netloc}/{rest}",
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate != image_url and candidate not in result:
            result.append(candidate)
    return result


def download_one_image(
    index: int,
    image_url: str,
    *,
    article_url: str,
    slug: str,
    data_dir: Path,
) -> dict[str, Any]:
    """下载、解密并保存一张图片。"""
    record: dict[str, Any] = {
        "source_url": image_url,
        "local_path": "",
        "bytes": 0,
        "sha256": "",
    }
    candidates = [image_url] + alternate_image_urls(image_url)
    last_error: Exception | None = None
    try:
        raw: bytes | None = None
        used_url = image_url
        for candidate in candidates:
            try:
                raw = request_bytes(
                    candidate,
                    referer=article_url,
                    retries=config.REQUEST_RETRIES if candidate == image_url else 1,
                )
                used_url = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if candidate != image_url:
                    logger.info("  候选图片地址也失败：%s（%s）", candidate, exc)
        if raw is None:
            raise last_error or CrawlError("没有可用的图片地址")

        image_bytes, ext = decrypt_image_bytes(raw)
        filename = f"{index:03d}.{ext}"
        relative_path = f"{config.IMAGES_DIRNAME}/{slug}/{filename}"
        output_path = data_dir / relative_path
        atomic_write_bytes(output_path, image_bytes)
        record.update(
            {
                "local_path": relative_path,
                "bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
        if used_url != image_url:
            record["downloaded_from"] = used_url
            logger.info("  图片 %03d 使用候选地址下载成功：%s", index, used_url)
        logger.info("  图片 %03d 已保存：%s（%d 字节）", index, relative_path, len(image_bytes))
        return record
    except Exception as exc:  # noqa: BLE001
        logger.warning("  图片 %03d 处理失败：%s（%s）", index, image_url, exc)
        record["error"] = str(exc)
        return record


def download_article_images(
    image_urls: list[str],
    *,
    article_url: str,
    slug: str,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """并发下载一篇文章的所有图片，返回 (成功记录, 失败记录)。"""
    if not image_urls:
        return [], []

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.DOWNLOAD_WORKERS)) as pool:
        futures = {
            pool.submit(
                download_one_image,
                index,
                image_url,
                article_url=article_url,
                slug=slug,
                data_dir=data_dir,
            ): index
            for index, image_url in enumerate(image_urls, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = {
                    "source_url": image_urls[index - 1],
                    "local_path": "",
                    "error": str(exc),
                }

    images: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    for index in range(1, len(image_urls) + 1):
        record = results.get(index)
        if not record:
            continue
        if record.get("local_path") and record.get("sha256"):
            if record["sha256"] in seen_sha:
                # 同一文章里重复的图片只保留一份
                try:
                    (data_dir / record["local_path"]).unlink()
                except OSError:
                    pass
                continue
            seen_sha.add(record["sha256"])
            images.append(record)
        else:
            errors.append(record)

    return images, errors


def process_article(
    article: dict[str, str],
    *,
    data_dir: Path,
    search_referer: str,
) -> dict[str, Any]:
    """抓取一篇文章的详情页并下载全部正文图片。"""
    url = article["url"]
    logger.info("抓取详情页：%s", url)
    session = requests.Session()
    html = request_text(session, url, referer=search_referer)
    title, published_at, author, image_urls = parse_detail_page(html, url)

    slug = article_slug(url)
    images, errors = download_article_images(
        image_urls,
        article_url=url,
        slug=slug,
        data_dir=data_dir,
    )

    entry = {
        "url": url,
        "title": title or article.get("title", ""),
        "published_at": published_at or article.get("published_at", ""),
        "author": author or article.get("author", ""),
        "slug": slug,
        "crawled_at": now_iso(),
        "status": "ok" if not errors else "partial",
        "image_count": len(images),
        "images": images,
        "image_errors": errors,
    }
    logger.info(
        "完成：%s，正文图片 %d 张，成功 %d 张，失败 %d 张",
        entry["title"][:40],
        len(image_urls),
        len(images),
        len(errors),
    )
    return entry


def entry_is_complete(entry: dict[str, Any], data_dir: Path) -> bool:
    """判断已有记录是否可以直接复用。"""
    if entry.get("status") != "ok":
        return False
    images = entry.get("images") or []
    for image in images:
        local_path = image.get("local_path")
        if not local_path or not (data_dir / local_path).is_file():
            return False
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_page_url(keyword: str, page: int) -> str:
    """构造搜索结果页地址（第 1 页没有页码后缀）。"""
    encoded = quote(keyword, safe="")
    base = config.SEARCH_URL_TEMPLATE.format(keyword=encoded)
    if page <= 1:
        return base
    return f"{base}{page}/"


def crawl(
    *,
    keyword: str,
    data_dir: Path,
    max_pages: int = 0,
    max_articles: int = 0,
    fresh: bool = False,
) -> dict[str, Any]:
    """执行爬取，返回完整的结果 JSON。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    results_path = data_dir / config.RESULTS_FILE.name

    existing_by_url: dict[str, dict[str, Any]] = {}
    if results_path.is_file() and not fresh:
        try:
            old_payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取已有 results.json 失败，将重新开始：%s", exc)
        else:
            old_keyword = old_payload.get("keyword")
            if old_keyword and old_keyword != keyword:
                raise SystemExit(
                    f"已有结果的关键词是「{old_keyword}」，与当前关键词「{keyword}」不同。\n"
                    "为避免数据混在一起，请加 --fresh 重新爬取，或修改 --data-dir。"
                )
            for entry in old_payload.get("articles", []):
                if entry.get("url"):
                    existing_by_url[entry["url"]] = entry

    session = requests.Session()
    search_url = build_page_url(keyword, 1)
    logger.info("搜索关键词：%s", keyword)
    logger.info("搜索地址：%s", search_url)

    collected: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    total_pages: int | None = None
    page = 1

    while True:
        if max_pages and page > max_pages:
            break
        if total_pages is not None and page > total_pages:
            break

        page_url = build_page_url(keyword, page)
        logger.info("抓取搜索结果第 %s 页：%s", page, page_url)
        try:
            html = request_text(session, page_url, referer=search_url)
        except CrawlError:
            if page == 1:
                raise
            logger.warning("第 %s 页抓取失败，停止翻页", page)
            break

        page_articles, page_total = parse_search_page(html, page_url)
        if page_total:
            total_pages = page_total
        if not page_articles:
            break

        for article in page_articles:
            if article["url"] in seen_urls:
                continue
            seen_urls.add(article["url"])
            collected.append(article)

        if max_articles and len(collected) >= max_articles:
            collected = collected[:max_articles]
            break

        page += 1
        time.sleep(config.REQUEST_INTERVAL)
        if page > 200:  # 防御性上限，避免异常站点无限翻页
            logger.warning("已达到 200 页防御上限，停止翻页")
            break

    logger.info("搜索结果共 %d 篇文章（站点总页数：%s）", len(collected), total_pages)

    output_articles: list[dict[str, Any]] = []
    for index, article in enumerate(collected, start=1):
        existing = existing_by_url.get(article["url"])
        if existing and entry_is_complete(existing, data_dir):
            logger.info("[%d/%d] 已抓取过，跳过：%s", index, len(collected), article["url"])
            output_articles.append(existing)
        else:
            logger.info("[%d/%d] 开始处理", index, len(collected))
            try:
                entry = process_article(
                    article, data_dir=data_dir, search_referer=search_url
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("文章抓取失败：%s（%s）", article["url"], exc)
                entry = {
                    "url": article["url"],
                    "title": article.get("title", ""),
                    "published_at": article.get("published_at", ""),
                    "author": article.get("author", ""),
                    "slug": article_slug(article["url"]),
                    "crawled_at": now_iso(),
                    "status": "failed",
                    "image_count": 0,
                    "images": [],
                    "image_errors": [{"source_url": "", "error": str(exc)}],
                }
            output_articles.append(entry)

        payload = {
            "keyword": keyword,
            "source_url": search_url,
            "base_url": config.BASE_URL,
            "crawled_at": now_iso(),
            "page_count": total_pages or page,
            "article_count": len(output_articles),
            "articles": output_articles,
        }
        atomic_write_json(results_path, payload)
        time.sleep(config.REQUEST_INTERVAL)

    payload = {
        "keyword": keyword,
        "source_url": search_url,
        "base_url": config.BASE_URL,
        "crawled_at": now_iso(),
        "page_count": total_pages or (page - 1 if total_pages is None else total_pages),
        "article_count": len(output_articles),
        "articles": output_articles,
    }
    atomic_write_json(results_path, payload)

    image_total = sum(len(item.get("images") or []) for item in output_articles)
    logger.info("爬取完成：文章 %d 篇，图片 %d 张", len(output_articles), image_total)
    logger.info("结果文件：%s", results_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="爬取每日大赛搜索结果中的详情页链接和正文图片"
    )
    parser.add_argument(
        "-k", "--keyword", default=config.SEARCH_KEYWORD, help="搜索关键词"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help=f"数据目录（默认：{config.DATA_DIR}）",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=config.MAX_PAGES,
        help="最多抓取多少页，0 表示全部",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=config.MAX_ARTICLES,
        help="最多抓取多少篇文章，0 表示不限制",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已有 results.json，重新爬取",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    crawl(
        keyword=args.keyword.strip(),
        data_dir=args.data_dir.expanduser().resolve(),
        max_pages=max(0, args.max_pages),
        max_articles=max(0, args.max_articles),
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
