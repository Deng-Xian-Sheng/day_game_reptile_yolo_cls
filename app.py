# -*- coding: utf-8 -*-
"""简单的 Web 网站：展示命中目标类别的链接、命中图片和其他图片。

用法：
    python app.py
    python app.py --host 0.0.0.0 --port 8000
    python app.py --data-dir /path/to/data

然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory, url_for

import config

logger = logging.getLogger("web")


def prepare_image(image: dict, data_dir: Path) -> dict:
    """给单张图片补充 URL 和文件是否存在标记。"""
    local_path = image.get("local_path") or ""
    image["url"] = url_for("serve_image", filename=local_path) if local_path else ""
    image["exists"] = bool(local_path) and (data_dir / local_path).is_file()
    return image


def prepare_matches(matches: list[dict], data_dir: Path) -> list[dict]:
    """给图片补充 URL 和文件是否存在标记，供模板使用。"""
    prepared: list[dict] = []
    for match in matches:
        if not isinstance(match, dict):
            continue

        images: list[dict] = []
        matched_images = match.get("matched_images") or []
        if not matched_images and match.get("matched_image"):
            matched_images = [match["matched_image"]]
        for image in matched_images:
            if isinstance(image, dict):
                images.append(prepare_image(image, data_dir))
        if not images:
            continue

        best = match.get("matched_image")
        if isinstance(best, dict):
            best = prepare_image(best, data_dir)
            if not best.get("local_path"):
                best = images[0]
            elif not any(
                image.get("local_path") == best.get("local_path") for image in images
            ):
                images.insert(0, best)
        else:
            best = images[0]

        match["matched_image"] = best
        match["matched_images"] = images

        other_images = []
        for image in match.get("other_images") or []:
            if isinstance(image, dict):
                other_images.append(prepare_image(image, data_dir))
        match["other_images"] = other_images

        # 排序用的数据
        match["image_total"] = len(images) + len(other_images)
        match["matched_count"] = len(images)
        match["other_count"] = len(other_images)
        match["best_confidence"] = float(best.get("confidence") or 0)
        match["published"] = match.get("published_at") or ""

        match["title"] = match.get("title") or "未命名内容"
        match["search_text"] = f"{match['title']} {match.get('url', '')}".lower()
        prepared.append(match)
    return prepared


def create_app(data_dir: Path) -> Flask:
    """创建 Flask 应用。"""
    data_dir = data_dir.expanduser().resolve()
    app = Flask(__name__)
    app.config["DATA_DIR"] = data_dir
    app.config["JSON_AS_ASCII"] = False

    @app.get("/")
    def index():
        matched_path = data_dir / config.MATCHED_FILE.name
        payload: dict = {}
        error = ""
        if matched_path.is_file():
            try:
                with matched_path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("读取 matched.json 失败：%s", exc)
                error = f"读取 {matched_path} 失败：{exc}"
        else:
            error = f"还没有分类结果：{matched_path}"

        matches = prepare_matches(payload.get("matches") or [], data_dir)
        stats = payload.get("stats") or {}
        return render_template(
            "index.html",
            payload=payload,
            matches=matches,
            stats=stats,
            error=error,
            matched_path=matched_path,
            data_dir=data_dir,
        )

    @app.get("/images/<path:filename>")
    def serve_image(filename: str):
        # 只允许访问 data/images/ 下的图片，同时防止 ../ 路径穿越
        images_root = (data_dir / config.IMAGES_DIRNAME).resolve()
        target = (data_dir / filename).resolve()
        if not target.is_relative_to(images_root) or target == images_root:
            abort(404)
        return send_from_directory(
            images_root, str(target.relative_to(images_root)), max_age=86400
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    return app


# 兼容 `flask --app app run` 的用法
app = create_app(config.DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="展示分类命中结果的 Web 网站")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help=f"数据目录（默认：{config.DATA_DIR}）",
    )
    parser.add_argument("--host", default=config.WEB_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=config.WEB_PORT, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    flask_app = create_app(args.data_dir)
    logger.info("网站地址：http://%s:%s", args.host, args.port)
    logger.info("数据目录：%s", args.data_dir.expanduser().resolve())
    flask_app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
