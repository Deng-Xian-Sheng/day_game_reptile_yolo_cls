# 每日大赛搜索结果爬取与图片分类

爬取指定搜索关键词下的详情页链接和正文图片，使用
[`likewendy/yolo26n-cls-porn`](https://huggingface.co/likewendy/yolo26n-cls-porn)
分类模型筛选出包含指定类别图片的链接，并用一个简单的 Web 网站展示：

- 每条链接
- 命中的图片（单独字段）
- 该链接的其他图片（单独字段）

## 快速开始

```bash
# 1. 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 登录 Hugging Face（模型是门控模型）
hf auth login

# 3. 修改 config.py 里的 SEARCH_KEYWORD 和 TARGET_CLASS

# 4. 爬取 -> 分类 -> 启动网站
python crawler.py
python classify.py
python app.py
```

浏览器打开 <http://127.0.0.1:8000>。

---

## 1. 目录结构

```text
.
├── config.py              # 搜索关键词、目标类别、站点/模型参数
├── crawler.py             # 爬取搜索结果、详情页和正文图片
├── classify.py            # 加载 HF 模型，筛选命中图片
├── app.py                 # Flask Web 网站
├── common.py              # 原子写文件等通用工具
├── static/app.js          # 搜索、排序、灯箱等前端逻辑
├── templates/index.html   # 网站页面（含 CSS，无外部 CDN）
├── tests/test_core.py     # 不依赖真实模型的核心逻辑测试
├── requirements.txt
├── data/                  # 运行后生成（已 gitignore）
│   ├── results.json       # 爬取结果
│   ├── matched.json       # 分类命中结果（网站读取）
│   ├── classify_cache.json# 推理缓存
│   └── images/<文章ID>/   # 解密后的本地图片
└── README.md
```

---

## 2. 安装

需要 Python 3.10 或更高版本。

### 2.1 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate

# 宿主机安装全部依赖（会安装 torch/ultralytics，体积较大）
pip install -r requirements.txt
```

如果宿主机只需要 CPU 推理，建议先装 CPU 版 torch，避免下载体积很大的 CUDA 依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

如果只想浏览已经生成好的 `data/matched.json`，可以只安装：

```bash
pip install flask
```

### 2.2 登录 Hugging Face

模型是门控模型，需要：

1. 登录有权限的 HF 账号；
2. 在模型页面点击同意访问条款；
3. 本机保存 token。

```bash
# huggingface_hub 新版 CLI
hf auth login
# 旧版本也可以使用：
# huggingface-cli login

# 或使用环境变量
export HF_TOKEN=hf_xxx
```

代码里不保存 token，`huggingface_hub` 会自动读取
`HF_TOKEN` 或 `~/.cache/huggingface/token`。首次运行分类脚本时会自动下载：

- `weights/best.onnx`（约 6 MB）
- `class35_conf_thresholds_by_name.json`
- `class35_zh_to_en.json`

### 2.3 虚拟机与宿主机

当前开发虚拟机只验证了爬虫、图片解密、mock 分类和网站；没有在虚拟机里跑
torch/ONNX 推理。真实模型推理请在宿主机执行。宿主机需要能访问
`huggingface.co` 和图片 CDN。

---

## 3. 需要你设置的变量

打开 `config.py`，最上面两个变量就是需要你设置的：

```python
# 搜索关键词
SEARCH_KEYWORD = "测试"

# 分类目标类别，中文或英文均可
TARGET_CLASS = "动漫"
```

也可以不改文件，用命令行参数临时覆盖：

```bash
python crawler.py --keyword "测试"
python classify.py --target "Anime"
```

其他常用参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BASE_URL` | `https://card.ksyytujr.cc` | 站点域名，换域名时改这里 |
| `MAX_PAGES` | `0` | 最多抓几页搜索结果，0 = 全部 |
| `MAX_ARTICLES` | `0` | 最多抓几篇文章，0 = 不限制 |
| `REQUEST_INTERVAL` | `0.6` | 页面请求间隔（秒） |
| `DOWNLOAD_WORKERS` | `4` | 每篇文章图片并发下载数 |
| `DATA_DIR` | `./data` | 数据目录，可用 `--data-dir` 覆盖 |
| `MODEL_DEVICE` | `cpu` | 推理设备；装了 onnxruntime-gpu 才改成 `0` |
| `MODEL_MARGIN` | `0.4` | top1-top2 置信度差门槛 |

> 站点域名可能变化。如果爬虫返回 404/403，先确认 `BASE_URL` 是否还是可用域名。

---

## 4. 使用流程

### 4.1 爬取

```bash
python crawler.py
```

常用调试命令：

```bash
# 只抓第 1 页、最多 3 篇文章
python crawler.py --max-pages 1 --max-articles 3

# 忽略已有结果，重新抓
python crawler.py --fresh

# 数据放到其他磁盘
python crawler.py --data-dir /mnt/data/mrds
```

爬取逻辑：

1. 从 `/search/<关键词>/` 开始翻页，读取全部搜索结果；
2. 进入每个 `/archives/<ID>/` 详情页；
3. 从 `.post-content` 中提取正文图片；
4. 图片真实地址在 `data-xkrkllgl` 属性中，`src` 是占位图；
5. 图片在 CDN 上是 AES-128-CBC 加密的，脚本会下载后解密；
6. 老文章图片地址可能多包一层原始 CDN 域名，脚本会自动尝试候选地址；
7. 解密后按文件头识别真实格式，保存为 `data/images/<文章ID>/001.jpg` 等；
8. 每处理完一篇文章就原子写入 `data/results.json`，中断后可继续。

断点续爬：再次执行 `python crawler.py` 时，状态为 `ok` 且本地图片齐全的文章会跳过；
状态为 `partial`（部分图片失败）或 `failed` 的文章会重试。

`data/results.json` 结构示例：

```json
{
  "keyword": "测试",
  "source_url": "https://card.ksyytujr.cc/search/%E6%B5%8B%E8%AF%95/",
  "base_url": "https://card.ksyytujr.cc",
  "crawled_at": "2026-10-21T17:27:09+08:00",
  "page_count": 3,
  "article_count": 2,
  "articles": [
    {
      "url": "https://card.ksyytujr.cc/archives/195902/",
      "title": "文章标题",
      "published_at": "2026-10-09T11:00:00+07:00",
      "author": "",
      "slug": "195902",
      "crawled_at": "2026-10-21T17:27:06+08:00",
      "status": "ok",
      "image_count": 9,
      "images": [
        {
          "source_url": "https://pic.sugfew.cn/upload_01/.../xxx.jpeg",
          "local_path": "images/195902/001.jpg",
          "bytes": 106594,
          "sha256": "..."
        }
      ],
      "image_errors": []
    }
  ]
}
```

### 4.2 分类筛选

```bash
python classify.py
```

常用参数：

```bash
# 指定目标类别（中文或英文）
python classify.py --target "动漫"
python classify.py --target "Anime"

# 有 CUDA 时
python classify.py --device 0

# 不使用缓存，强制重新推理
python classify.py --no-cache

# 只分析前 20 张图片，快速验证流程
python classify.py --max-images 20

# 没有安装模型依赖时，用假分类器测试网站
python classify.py --mock
```

推理规则与模型卡一致：

```python
top1_name == 目标类别
and top1_conf >= thresholds[目标类别]
and (top1_conf - top2_conf) >= 0.4
```

其中每类阈值来自模型仓库的 `class35_conf_thresholds_by_name.json`。
`top5` 由 Ultralytics 按置信度降序返回。

输出 `data/matched.json`：

```json
{
  "keyword": "测试",
  "target_class": "动漫",
  "target_class_en": "Anime",
  "model": "likewendy/yolo26n-cls-porn",
  "criteria": {
    "top1_must_equal_target": true,
    "confidence_threshold": 0.5,
    "margin": 0.4,
    "margin_definition": "top1_conf - top2_conf"
  },
  "stats": {
    "articles": 72,
    "images": 648,
    "matched_articles": 12,
    "matched_images": 25
  },
  "matches": [
    {
      "url": "https://card.ksyytujr.cc/archives/195902/",
      "title": "文章标题",
      "published_at": "2026-10-09T11:00:00+07:00",
      "match_count": 1,
      "matched_image": {
        "local_path": "images/195902/003.jpg",
        "source_url": "https://pic.sugfew.cn/.../xxx.jpeg",
        "predicted_class": "动漫",
        "confidence": 0.95,
        "second_class": "口交",
        "second_confidence": 0.02,
        "margin": 0.93,
        "top5": [
          {"class": "动漫", "confidence": 0.95},
          {"class": "口交", "confidence": 0.02}
        ]
      },
      "matched_images": [
        {
          "local_path": "images/195902/003.jpg",
          "source_url": "https://pic.sugfew.cn/.../xxx.jpeg",
          "predicted_class": "动漫",
          "confidence": 0.95
        }
      ],
      "other_images": [
        {
          "local_path": "images/195902/001.jpg",
          "source_url": "https://pic.sugfew.cn/.../yyy.jpeg",
          "predicted_class": "口交",
          "confidence": 0.71
        }
      ]
    }
  ]
}
```

字段说明：

- `matched_image`：该链接中置信度最高的命中图片；
- `matched_images`：该链接所有命中图片（按置信度降序，包含 `matched_image`）；
- `other_images`：该链接中未命中的其他图片；
- 一篇链接只生成一条记录，避免同一链接重复出现。

推理缓存保存在 `data/classify_cache.json`，按图片大小和修改时间复用；
换模型或改目标类别不会误用旧的判定，目标类别是在缓存之后重新计算的。

### 4.3 启动网站

```bash
python app.py
```

浏览器打开 <http://127.0.0.1:8000>。

局域网/宿主机访问：

```bash
python app.py --host 0.0.0.0 --port 8000
```

网站功能：

- 展示链接、标题、日期、命中图片；
- 命中图片大图预览（点击图片打开灯箱）；
- 同链接其他命中图片缩略图；
- 同链接其他图片缩略图；
- 按标题/链接实时搜索，支持包含、排除及组合条件；
- 多字段、多条件排序，可自由组合；
- 图片懒加载，手机端自适应；
- 直接点击「打开原链接」跳转。

#### 搜索语法

搜索范围是「标题 + 原链接」，默认不区分大小写。

| 写法 | 含义 |
| --- | --- |
| `动漫 护士` | 同时包含「动漫」和「护士」 |
| `包含动漫和护士` | 同时包含「动漫」和「护士」 |
| `动漫*护士` | 同时包含「动漫」和「护士」，`*` 表示两者之间可有任意字符 |
| `动漫 -口交` | 包含「动漫」，且不包含「口交」 |
| `!口交` | 不包含「口交」 |
| `不包含口交和广告` | 同时排除「口交」「广告」 |
| `包含动漫和护士 不包含口交` | 包含条件与排除条件组合 |
| `"动漫 护士"` | 引号内视为一个完整词组 |

说明：

- 多个包含词用空格、逗号、`*`、`和`、`与`、`及` 分隔，关键词在标题中**不要求连续**；
  例如 `关键词1 关键词2` 或 `关键词1*关键词2` 都可以匹配「哈哈关键词1哈哈哈关键词2哈哈」。
- 排除词支持 `-词`、`!词`、`不包含词`、`不含词`、`排除词`。
- 包含和排除可以混合使用，也可以先写包含再写排除，或反过来。
- 关键词本身含有「和 / 与 / 及」时，建议用引号括起来，例如 `"和平"`。
- 输入框下方会实时显示解析出的「包含 / 排除」标签，方便确认语法是否符合预期。

#### 排序

可排序字段：

- 发布时间 `published_at`
- 总图片数（命中图片 + 其他图片）
- 命中图片数
- 其他图片数
- 最高置信度（最佳命中图片）

使用方法：

1. 选择字段和升序/降序；
2. 点击「添加排序」；
3. 可继续添加多个条件，标签从左到右就是优先级顺序；
4. 点击排序标签可切换升序/降序，点击 `×` 移除该条件；
5. 点击「清空」恢复 `matched.json` 的原始顺序（分类结果默认按置信度降序）。

例如：先添加「发布时间 降序」，再添加「最高置信度 降序」，表示先按时间排序，
时间相同时再按置信度排序。搜索和排序可以同时使用；缺失发布时间的内容始终排在最后。

默认只监听 `127.0.0.1`。内容属于成人内容，建议只在本机或可信内网访问；
如果使用 `--host 0.0.0.0` 暴露到局域网，请自行加访问控制。

网站每次请求都会重新读取 `data/matched.json`，重新跑完分类后刷新页面即可。
图片通过 `/images/images/<文章ID>/<文件名>` 提供，Flask 的
`send_from_directory` 会阻止路径穿越。

---

## 5. 模型类别

共 35 类，中文和英文都可用于 `TARGET_CLASS` / `--target`：

| 中文 | 英文 | 中文 | 英文 |
| --- | --- | --- | --- |
| AI生成 | AI Generated | 乳交 | Titfuck |
| 偷窥 | Voyeurism | 公共暴露 | Public Exposure |
| 内射 | Creampie | 动漫 | Anime |
| 卷发 | Curly Hair | 双插 | Double Penetration |
| 口交 | Blowjob | 口爆大会 | Bukkake |
| 圣诞 | Christmas | 女仆 | Maid |
| 女女假阳具 | Lesbian Strapon | 女女摩擦 | Lesbian Tribbing |
| 女着男裸 | CFNM | 孕妇 | Pregnant |
| 恋尿 | Pissing | 手交 | Handjob |
| 接吻 | Kissing | 油亮 | Oiled |
| 潮吹 | Squirting | 男性独处 | Solo Male |
| 男性群像 | Male Group | 皮革乳胶 | Leather and Latex |
| 纹身 | Tattoo | 绿帽视角 | Cuckold |
| 肛门特写 | Asshole Close-up | 脚交 | Footjob |
| 腋窝 | Armpit | 舔肛 | Rimjob |
| 舔阴 | Pussy Licking | 裤装 | Pants |
| 跨种族 | Interracial | 镜前 | Mirror |
| 骑脸 | Facesitting | | |

模型在验证集上的 Top-1 约为 67.97%，Top-5 约为 90.70%。单个类别也有自己的
精确率和阈值，因此命中数量少是正常的。

模型许可证为 GPL-3.0；下载和使用模型时请遵守模型页面的许可条款。

---

## 6. 测试

不需要真实模型即可运行核心逻辑测试：

```bash
python -m unittest discover -s tests -v
```

已覆盖：

- 本站 AES 图片解密、未加密图片透传；
- 搜索结果页解析；
- 详情页图片提取（优先 `data-xkrkllgl`，过滤占位图）；
- top-1 / 阈值 / margin 判定。

流程联调（不加载真实模型）：

```bash
# 1. 先抓少量真实图片
python crawler.py --max-pages 1 --max-articles 2 --data-dir /tmp/mrds-test

# 2. 用 mock 分类器生成 matched.json
python classify.py --data-dir /tmp/mrds-test --target 动漫 --mock

# 3. 启动网站查看
python app.py --data-dir /tmp/mrds-test --port 8001
```

`--mock` 会按图片内容哈希让约 1/5 的图片命中目标类别，仅用于验证 JSON 结构、
网站展示和流程，不代表真实模型结果。

开发时实测（2026-10-21）：搜索「测试」第 1 页共 30 篇文章、369 张正文图片，
其中 368 张成功下载并解密，1 张源地址本身已失效（站点数据问题）。

---

## 7. 常见问题

### 7.1 爬虫 403 / 404 / 返回空列表

- 确认 `config.BASE_URL` 是当前可用域名；
- 用浏览器打开一次搜索页，确认页面结构没有改版；
- 适当增大 `REQUEST_INTERVAL`，避免请求过快；
- 如果站点新增了 Cookie/Cloudflare 校验，需要额外处理（当前站点不需要）。

### 7.2 图片下载成功但无法解密

错误会记录在 `results.json` 的 `image_errors` 里，常见原因：

- 站点更换了 AES key/iv：更新 `config.IMAGE_AES_KEY` 和 `IMAGE_AES_IV`；
- 图片地址已失效；
- 图片不是本站加密格式，脚本会先按文件头识别，普通图片会直接保存。

### 7.3 分类模型下载失败

- `hf auth login` 是否成功；
- 是否已在模型页面接受访问条款；
- 网络是否可访问 `huggingface.co`；
- 也可以用 `--model-dir` 指向本地目录，目录内放
  `best.onnx`、`class35_conf_thresholds_by_name.json`、
  `class35_zh_to_en.json`。

### 7.4 ultralytics 报任务类型错误

代码已经显式使用：

```python
YOLO(model_path, task="classify")
```

如果仍报错，尝试升级：

```bash
pip install -U ultralytics onnxruntime
```

模型是 YOLO26n-cls 导出的 ONNX，需要较新的 ultralytics 才能正确加载。

### 7.5 命中结果太少

- 模型 Top-1 约 68%，本身会漏判；
- 每类阈值不同，部分类别阈值较高；
- `margin=0.4` 是模型卡建议值，降低它会增加误判；
- 可先用 `--mock` 确认流程，再用真实模型观察 `matched.json` 里的
  `top5` 和 `checks` 字段，判断是类别不对、置信度不够还是 margin 不够。

---

## 8. 备注

- 本项目只抓取搜索结果中公开可见的页面和图片，请遵守目标站点条款和当地法律；
- 图片总体积取决于关键词和文章数量，建议先 `--max-articles` 小规模试跑；
- 代码注释、网站文案均为中文；
- 站点改版时主要需要调整：`BASE_URL`、`parse_search_page`、`parse_detail_page`、
  以及图片 AES 参数。
