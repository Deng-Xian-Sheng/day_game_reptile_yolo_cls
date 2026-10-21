# -*- coding: utf-8 -*-
"""不依赖真实模型的核心逻辑测试。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import config
from classify import YoloClassifier, decide
from crawler import (
    alternate_image_urls,
    decrypt_image_bytes,
    parse_detail_page,
    parse_search_page,
)


class DecryptImageTest(unittest.TestCase):
    def test_encrypted_jpeg(self) -> None:
        # 伪造一个 JPEG 文件头，按站点相同的 AES-CBC + PKCS7 加密
        plain = b"\xff\xd8\xff\xe0" + b"0" * 60
        padder = padding.PKCS7(128).padder()
        padded = padder.update(plain) + padder.finalize()
        encryptor = Cipher(
            algorithms.AES(config.IMAGE_AES_KEY),
            modes.CBC(config.IMAGE_AES_IV),
        ).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()

        decrypted, ext = decrypt_image_bytes(encrypted)
        self.assertEqual(ext, "jpg")
        self.assertTrue(decrypted.startswith(b"\xff\xd8\xff"))

    def test_plain_image_passthrough(self) -> None:
        plain = b"\x89PNG\r\n\x1a\n" + b"0" * 32
        decrypted, ext = decrypt_image_bytes(plain)
        self.assertEqual(ext, "png")
        self.assertEqual(decrypted, plain)


class ParseHtmlTest(unittest.TestCase):
    def test_alternate_image_urls(self) -> None:
        urls = alternate_image_urls(
            "https://pic.sugfew.cn/pic.old-cdn.cn/upload_01/x/y.jpeg"
        )
        self.assertEqual(
            urls,
            [
                "https://pic.old-cdn.cn/upload_01/x/y.jpeg",
                "https://pic.sugfew.cn/upload_01/x/y.jpeg",
            ],
        )
        self.assertEqual(
            alternate_image_urls("https://pic.sugfew.cn/upload_01/x/y.jpeg"), []
        )

    def test_parse_search_page(self) -> None:
        html = """
        <html><body>
          <article>
            <a href="/archives/123/"><div class="post-card"></div></a>
            <h2 class="post-card-title">测试标题</h2>
            <meta itemprop="datePublished" content="2026-01-02T03:04:05+08:00">
          </article>
          <div class="page-info">1/3</div>
        </body></html>
        """
        articles, total_pages = parse_search_page(html, "https://example.com/search/x/")
        self.assertEqual(total_pages, 3)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["url"], "https://example.com/archives/123/")
        self.assertEqual(articles[0]["title"], "测试标题")

    def test_parse_detail_page(self) -> None:
        html = """
        <html><body>
          <h1>文章标题</h1>
          <meta itemprop="datePublished" content="2026-01-02T03:04:05+08:00">
          <div class="post-content">
            <p><img src="/usr/plugins/tbxw/zw.png"
                    data-xkrkllgl="https://pic.example.com/a.jpeg"></p>
            <p><img src="https://pic.example.com/plain.png"></p>
            <p><img src="/usr/themes/x/banner.png"></p>
          </div>
        </body></html>
        """
        title, published, _author, images = parse_detail_page(
            html, "https://example.com/archives/123/"
        )
        self.assertEqual(title, "文章标题")
        self.assertEqual(published, "2026-01-02T03:04:05+08:00")
        self.assertEqual(
            images,
            ["https://pic.example.com/a.jpeg", "https://pic.example.com/plain.png"],
        )


class DecideTest(unittest.TestCase):
    def test_top5_mapping_and_sort(self) -> None:
        class FakeProbs:
            top5 = [2, 0, 1]
            top5conf = [0.70, 0.20, 0.10]

        class FakeResult:
            probs = FakeProbs()

        class FakeModel:
            names = {0: "AI生成", 1: "乳交", 2: "Anime"}

        classifier = object.__new__(YoloClassifier)
        classifier.model = FakeModel()
        classifier.thresholds = {"AI生成": 0.5, "乳交": 0.5, "动漫": 0.5}
        classifier.en2zh = {"anime": "动漫"}

        top5 = classifier._result_to_top5(FakeResult())
        # 英文类名会映射为中文，且按置信度降序排列
        self.assertEqual(top5[0], ("动漫", 0.70))
        self.assertEqual(top5[1], ("AI生成", 0.20))
        self.assertEqual(top5[2], ("乳交", 0.10))

    def test_match(self) -> None:
        result = decide(
            [("动漫", 0.90), ("其他", 0.10)],
            target_zh="动漫",
            thresholds={"动漫": 0.5},
            margin=0.4,
        )
        self.assertTrue(result["matched"])
        self.assertTrue(result["checks"]["margin_ok"])

    def test_top1_not_target(self) -> None:
        result = decide(
            [("其他", 0.90), ("动漫", 0.05)],
            target_zh="动漫",
            thresholds={"动漫": 0.5},
            margin=0.4,
        )
        self.assertFalse(result["matched"])

    def test_margin_not_enough(self) -> None:
        result = decide(
            [("动漫", 0.70), ("其他", 0.55)],
            target_zh="动漫",
            thresholds={"动漫": 0.5},
            margin=0.4,
        )
        self.assertFalse(result["matched"])
        self.assertFalse(result["checks"]["margin_ok"])


if __name__ == "__main__":
    unittest.main()
