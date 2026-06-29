# 审计知识问答系统 (Audit Knowledge Base QA)

基于 RAG（检索增强生成）的审计领域知识库问答系统，支持多格式文档接入、混合检索、流式问答、敏感词审计与提示词防注入。

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [部署](#部署)
- [常见问题](#常见问题)
- [许可协议](#许可协议)

## 功能特性

### 知识问答
- **流式 / 同步双模式**：支持 SSE 流式输出与一次性返回，通过 `stream` 参数切换
- **多轮对话**：基于 Redis 的会话管理，支持会话超时与上下文续接
- **意图识别**：自动判断是否需要开启新会话，避免上下文串扰
- **引用溯源**：返回回答所依据的源文档信息
- **高频问题缓存**：基于语义相似度命中缓存，降低重复检索成本

### 文档处理
- **多格式支持**：PDF（数字版 / 扫描件）、DOCX，通过 Apache Tika 解析
- **OCR 能力**：完整版集成 PaddleOCR，支持扫描件 PDF 文字识别
- **智能分块**：基于 `RecursiveCharacterTextSplitter` 的递归字符分块
- **对象存储接入**：从 MinIO 拉取文档，处理后写入 Milvus 向量库
- **增量维护**：支持文档新增（`NEED_AI`）与删除（`NO_AI`）操作

### 检索增强
- **混合检索**：支持 `vector`（向量）、`keyword`（关键词）、`hybrid`（混合）三种模式
- **重排序**：可选 Rerank 模型对召回结果二次排序，提升相关性
- **上下文压缩**：基于 tiktoken 的 token 计数与上下文裁剪，控制 LLM 输入长度

### 安全护栏
- **敏感词审计**：基于 Trie 树的敏感词检测，结合语义分析降低误报（技术语境识别）
- **提示词防注入**：检测用户输入中的提示词注入攻击模式
- **白名单机制**：支持白名单词与动态词库更新（Redis + 本地文件双存储）

## 系统架构

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│   Client     │────▶│              FastAPI (server.py)             │
└──────────────┘     │  ┌────────────┐  ┌─────────────────────────┐ │
                     │  │ 安全护栏    │  │      RAG 系统            │ │
                     │  │ SecurityGuard│ │   ┌──────────────────┐  │ │
                     │  │ - 敏感词检测 │  │   │ 检索器 Retriever  │  │ │
                     │  │ - 注入防御   │  │   │ 排序器 Ranker     │  │ │
                     │  └────────────┘  │   │ 上下文压缩        │  │ │
                     │                  │   │ LLM 客户端         │  │ │
                     │  ┌────────────┐  │   └──────────────────┘  │ │
                     │  │ 会话/上下文 │  └─────────────────────────┘ │
                     │  │  管理器     │  ┌─────────────────────────┐ │
                     │  └────────────┘  │     文档处理器           │ │
                     │                  │  MinIO → 分块 → Milvus   │ │
                     └──────────────────┴─────────────────────────┘
                              │                    │            │
                   ┌──────────┴──────┐   ┌─────────┴──┐   ┌─────┴────┐
                   │     Redis       │   │   Milvus   │   │  MinIO   │
                   │ 会话/缓存/词库   │   │  向量数据库 │   │ 对象存储  │
                   └─────────────────┘   └────────────┘   └──────────┘
```

**核心数据流**：

1. **建库流程**：MinIO 文档 → Tika/PDF/DOCX 解析 → 文本分块 → Embedding → Milvus 向量入库
2. **问答流程**：用户输入 → 安全护栏检查 → 意图识别 → 混合检索 → 重排序 → 上下文压缩 → LLM 生成 → 流式返回

## 技术栈

| 分类 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| RAG 框架 | LangChain（LCEL 链）、langchain-milvus、langchain-openai |
| 大模型 | OpenAI 兼容接口（LLM / Embedding / Rerank） |
| 向量数据库 | Milvus |
| 对象存储 | MinIO |
| 缓存与会话 | Redis |
| 文档解析 | Apache Tika、pdfplumber、python-docx |
| OCR | PaddleOCR + PaddlePaddle（完整版） |
| 配置管理 | pydantic-settings + python-dotenv |

## 项目结构

```
kbqa/
├── configs/                  # 配置模块
│   └── config.py             # 基于 pydantic-settings 的统一配置
├── core/                     # 核心模块
│   ├── models/               # 模型客户端
│   │   ├── embeddings.py     # Embedding 模型
│   │   ├── llm.py            # LLM 客户端（同步/流式）
│   │   └── rerank.py         # Rerank 重排序模型
│   ├── indexing.py           # 文档索引（MinIO → Milvus）
│   └── querying.py           # RAG 查询链（LCEL）
├── helps/                    # 辅助模块
│   ├── chunk_processor.py    # 文本分块处理
│   ├── context_compression.py# 上下文压缩（token 计数与裁剪）
│   ├── docx_processor.py     # DOCX 文档处理
│   ├── intent_recognition.py # 意图识别（新会话判定）
│   ├── pdf_processor.py      # PDF 文档处理（含 OCR）
│   ├── pdf_processor_light.py# PDF 文档处理（轻量版）
│   ├── question_cache.py     # 高频问题缓存
│   ├── ranker.py             # 排序器
│   └── retriever.py          # 检索器（vector/keyword/hybrid）
├── utils/                    # 工具模块
│   ├── context_manager.py    # 上下文管理
│   ├── conversation_manager.py# 对话会话管理（Redis）
│   ├── logger.py             # 日志管理
│   ├── milvus_manager.py     # Milvus 向量库管理
│   ├── minio_manager.py      # MinIO 对象存储管理
│   ├── prompt_manager.py     # 提示词模板管理
│   ├── redis_manager.py      # Redis 连接管理
│   └── security_guard.py     # 安全护栏（敏感词 + 注入防御）
├── docs/                     # 资源文件
│   ├── Vocabulary/           # 敏感词库目录（含示例词库，可按需替换）
│   ├── stopword.txt          # 停用词
│   └── whitelist.txt         # 敏感词白名单
├── logs/                     # 运行日志（运行时生成）
├── server.py                 # FastAPI 服务入口
├── requirements.txt          # 完整版依赖
├── requirements-light.txt    # 轻量版依赖（无 OCR）
├── Dockerfile.full           # 完整版镜像（含 OCR）
├── Dockerfile.light          # 轻量版镜像
├── deploy.sh                 # 部署脚本
├── .env                      # 环境变量（需自行创建，不提交）
├── .gitignore
└── README.md
```

## 环境准备

### 外部依赖服务

| 服务 | 默认地址 | 用途 |
|------|----------|------|
| Milvus | `localhost:19530` | 向量数据库 |
| MinIO | `localhost:9000` | 文档对象存储 |
| Redis | `localhost:6379` | 会话 / 缓存 / 敏感词库 |
| Apache Tika Server | `http://localhost:9998` | 文档解析服务 |
| OpenAI 兼容 LLM 服务 | 自定义 | LLM / Embedding / Rerank |

### Python 环境

- Python 3.11+

## 快速开始

1. **克隆仓库**

   ```bash
   git clone <repo-url>
   cd kbqa
   ```

2. **安装依赖**

   ```bash
   # 完整版（含 OCR，支持扫描件）
   pip install -r requirements.txt

   # 轻量版（无 OCR，镜像更小）
   pip install -r requirements-light.txt
   ```

3. **配置环境变量**

   复制示例配置并填入实际值：

   ```bash
   cp .env.example .env
   ```

   各项含义参考下方 [配置说明](#配置说明)。

4. **准备敏感词库（可选）**

   若需启用敏感词审计，将敏感词库文件放入 `docs/Vocabulary/` 目录，每个文件一行一个词。
   词库也会优先从 Redis 加载，本地文件作为备份。

5. **启动服务**

   ```bash
   python server.py
   # 或
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

6. **验证服务**

   ```bash
   curl http://localhost:8000/health
   ```

## 配置说明

所有配置通过环境变量（`.env` 文件）注入，由 `configs/config.py` 统一读取。关键配置项如下：

### 服务配置
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HOST` | 监听地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `8000` |
| `CALLBACK_URL` | 文档处理完成后的回调地址 | 空 |

### LLM 配置
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_URL` | LLM 服务地址 | 空 |
| `API_KEY` | LLM API Key | 空 |
| `LLM_MODEL` | 模型名称 | `gpt-4` |
| `LLM_TEMPERATURE` | 采样温度 | `0.1` |
| `LLM_MAX_TOKENS` | 最大生成 token | `2048` |
| `LLM_CONTEXT_LENGTH` | 模型上下文长度 | `32768` |

### Embedding / Rerank 配置
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EMBEDDING_API_URL` | Embedding 服务地址 | 空 |
| `EMBEDDING_MODEL_NAME` | Embedding 模型名 | `text-embedding-3-small` |
| `RERANK_API_URL` | Rerank 服务地址 | 空 |
| `RERANK_MODEL_NAME` | Rerank 模型名 | `rerank-english-v2.0` |

### 向量库 / 对象存储 / 缓存
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus 地址 | `localhost` / `19530` |
| `MILVUS_USER` / `MILVUS_PASSWORD` | Milvus 认证 | 空 |
| `MILVUS_COLLECTION_NAME` | 集合名 | `documents` |
| `MILVUS_DIMENSION` | 向量维度 | `1536` |
| `MINIO_ENDPOINT` | MinIO 地址 | `localhost:9000` |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO 凭证 | 需配置 |
| `MINIO_BUCKET_NAME` | 桶名 | `documents` |
| `REDIS_HOST` / `REDIS_PORT` | Redis 地址 | `localhost` / `6379` |
| `REDIS_PASSWORD` | Redis 密码 | 空 |
| `CONVERSATION_TIMEOUT` | 会话超时（秒） | `1800` |

### 检索与处理策略
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RETRIEVAL_MODE` | 检索方式：`vector` / `keyword` / `hybrid` | `hybrid` |
| `ENABLE_RE_RANKING` | 是否启用重排序 | `True` |
| `ENABLE_CONTEXT_COMPRESSION` | 是否启用上下文压缩 | `True` |
| `MAX_CONTEXT_TOKENS` | 最大上下文 token | `8000` |
| `CHUNKING_STRATEGY` | 分块策略 | `generic` |
| `PDF_PROCESSOR_MODE` | PDF 处理模式：`auto` / `digital` / `scanned` | `auto` |
| `ENABLE_OCR` | 是否启用 OCR | `False` |
| `TIKA_SERVER_URL` | Tika Server 地址 | 需配置 |
| `STREAM_CHUNK_SIZE` | 流式输出分块大小 | `50` |

### `.env` 示例

```ini
# 服务
HOST=0.0.0.0
PORT=8000

# LLM
BASE_URL=https://your-llm-endpoint/v1
API_KEY=your-api-key
LLM_MODEL=gpt-4

# Embedding
EMBEDDING_API_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Tika
TIKA_SERVER_URL=http://localhost:9998
```

## API 接口

### 1. 知识文档预处理

**POST** `/api/v1/log_audit/ai/kbqa/kb`

将 MinIO 中的文档解析、分块、向量化后写入 Milvus。

**请求体**：

```json
{
  "id": "请求唯一ID",
  "data": [
    {
      "file_id": "文档ID",
      "file_path": "MinIO中的文档路径",
      "file_label": "NEED_AI"
    }
  ]
}
```

- `file_label`：
  - `NEED_AI`：解析并入库
  - `NO_AI`：从向量库中删除该文档

**响应**：

```json
{
  "id": "请求唯一ID",
  "code": 200,
  "message": "处理成功，共处理N个文件"
}
```

### 2. 知识问答

**POST** `/api/v1/log_audit/ai/kbqa/qa`

支持 SSE 流式输出与一次性返回。

**请求体**：

```json
{
  "id": "请求唯一ID",
  "user_id": "用户ID",
  "user_input": "用户问题",
  "chat_id": "对话ID",
  "chat_topic": "对话主题",
  "session_id": "会话ID",
  "skill": "knowledge_qa",
  "stream": true,
  "history_messages": []
}
```

- `skill` 固定取值 `knowledge_qa`
- `stream` 为 `true` 时返回 SSE 流，每个事件格式：

```
data: {"id":"...","code":200,"message":"成功","chat_id":"...","session_id":"...","data":{"final_answer":"...","status":"PROCESSING","reference":null}}

data: [DONE]
```

- 命中敏感词返回 `code: 400`
- 会话过期返回 `code: 419` 并下发新的 `session_id`

### 3. 健康检查

**GET** `/health`

## 部署

### Docker 部署

项目提供两套 Dockerfile：

- **`Dockerfile.full`**：完整版，含 OpenCV / PaddleOCR，支持扫描件 OCR，镜像较大
- **`Dockerfile.light`**：轻量版，仅处理数字版 PDF，镜像较小

```bash
# 完整版
docker build -f Dockerfile.full -t kbqa:full .
docker run -d -p 8000:8000 --env-file .env kbqa:full

# 轻量版
docker build -f Dockerfile.light -t kbqa:light .
docker run -d -p 8000:8000 --env-file .env kbqa:light
```

### 脚本部署

```bash
bash deploy.sh
```

### 版本选择建议

| 场景 | 推荐版本 |
|------|----------|
| 仅处理数字版 PDF / DOCX | 轻量版 |
| 需处理扫描件 PDF / 图片 | 完整版（`ENABLE_OCR=true`） |
| 容器镜像体积敏感 | 轻量版 |

## 常见问题

**Q: 启动时提示「敏感词库为空」？**
A: 敏感词库优先从 Redis 加载，其次从 `docs/Vocabulary/` 目录加载。两者均为空时将使用空敏感词列表，敏感词审计功能不生效，但不影响问答。请将敏感词文件放入 `docs/Vocabulary/` 或写入 Redis。

**Q: 向量维度报错？**
A: `MILVUS_DIMENSION` 必须与 `EMBEDDING_MODEL_NAME` 实际输出的向量维度一致，否则入库失败。

**Q: 如何切换检索模式？**
A: 修改 `.env` 中的 `RETRIEVAL_MODE` 为 `vector` / `keyword` / `hybrid`。

**Q: OCR 启用后镜像构建失败？**
A: PaddleOCR / PaddlePaddle 体积较大，确保使用 `Dockerfile.full` 并有充足磁盘空间；或使用轻量版镜像并关闭 OCR。

**Q: 会话过期如何处理？**
A: 会话默认 30 分钟（`CONVERSATION_TIMEOUT`）过期，过期后接口返回 `419` 并下发新的 `session_id`，客户端需使用新 ID 重新发起对话。

## 许可协议

本项目仅供学习交流使用。使用前请遵守所接入的大模型、向量数据库等第三方服务的相关许可与服务条款。
