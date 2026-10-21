# -*- coding: utf-8 -*-
"""全局配置。

只需要优先修改「需要你设置的变量」一节；其余参数一般不用动。
"""

from pathlib import Path

# ============================================================================
# 需要你设置的变量
# ============================================================================

# 1) 搜索关键词：爬取该关键词的全部搜索结果
SEARCH_KEYWORD = "测试"

# 2) 分类模型的目标类别：图片被模型判为该类别时才会进入 matched.json
#    支持中文或英文，例如 "动漫" / "Anime"、"口交" / "Blowjob"
#    全部可选类别见 README.md 的「模型类别」一节
TARGET_CLASS = "动漫"

# ============================================================================
# 站点与爬取参数
# ============================================================================

BASE_URL = "https://card.ksyytujr.cc"
SEARCH_PATH_TEMPLATE = "/search/{keyword}/"

# 最多抓取多少个搜索结果页，0 表示全部
MAX_PAGES = 0
# 最多抓取多少篇文章，0 表示不限制（调试时可设为 3）
MAX_ARTICLES = 0

REQUEST_TIMEOUT = (15, 60)      # (连接超时, 读取超时)，单位秒
REQUEST_INTERVAL = 0.6          # 页面请求之间的间隔，避免请求过快
REQUEST_RETRIES = 3             # 请求失败重试次数
DOWNLOAD_WORKERS = 4            # 每篇文章内并发下载图片的线程数
MAX_IMAGE_BYTES = 30 * 1024 * 1024

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# 站点图片加密参数（来自前端 JS 的 AES-CBC key/iv，使用 UTF-8 字节）
# 如果站点以后更换了加密参数，只需修改这里。
IMAGE_AES_KEY = b"f5d965df75336270"
IMAGE_AES_IV = b"97b60394abc2fbe1"

# ============================================================================
# 目录与文件
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 爬取结果（详情页链接 + 本地图片路径）
RESULTS_FILE = DATA_DIR / "results.json"
# 分类命中的结果（Web 网站读取这个文件）
MATCHED_FILE = DATA_DIR / "matched.json"
# 本地图片相对数据目录的子目录名
IMAGES_DIRNAME = "images"

# 搜索结果页的地址模板，例如 https://card.ksyytujr.cc/search/测试/
SEARCH_URL_TEMPLATE = BASE_URL.rstrip("/") + SEARCH_PATH_TEMPLATE

# ============================================================================
# 分类模型参数
# ============================================================================

MODEL_REPO_ID = "likewendy/yolo26n-cls-porn"
MODEL_ONNX_FILE = "weights/best.onnx"
MODEL_THRESHOLDS_FILE = "class35_conf_thresholds_by_name.json"
MODEL_ZH2EN_FILE = "class35_zh_to_en.json"

MODEL_IMG_SIZE = 224
MODEL_DEVICE = "cpu"            # 默认 CPU；仅在安装了 onnxruntime-gpu 时改 "0"
MODEL_BATCH_SIZE = 16
# 模型卡建议的 top1-top2 置信度差值门槛
MODEL_MARGIN = 0.4

# 分类结果缓存，避免重复推理；文件不存在时会自动创建
CLASSIFY_CACHE_FILE = DATA_DIR / "classify_cache.json"

# ============================================================================
# Web 网站参数
# ============================================================================

WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
