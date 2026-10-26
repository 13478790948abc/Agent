# 智能客服 RAG 知识库问答系统

基于 **Streamlit + LangChain + Chroma + 阿里云百炼（DashScope）** 搭建的服装电商智能客服示例项目。

系统把商品相关的说明文档（尺码推荐、洗涤养护、颜色选择）切分成文本块并向量化存入 Chroma，用户提问时先做相似度检索，再把召回内容作为参考资料交给大模型生成回答，同时保留多轮对话历史。

## 功能

- **知识库上传**：网页上传 TXT 文件，自动去重（MD5）、分块、向量化并写入 Chroma。
- **知识库问答**：检索 Top-2 相关片段作为上下文，流式输出回答，支持多轮对话。
- **本地持久化**：向量库落在 `chroma_db/`，会话历史按 `session_id` 落在 `chat_history/`，避免同一内容重复入库。

## 技术栈

| 组件 | 说明 |
| --- | --- |
| Streamlit | Web 界面与流式输出 |
| LangChain | LCEL 链式编排、Prompt、消息历史 |
| Chroma | 本地向量数据库 |
| DashScope | `text-embedding-v4` 向量化，`qwen3.8-max` 对话模型 |

## 目录结构

```
RAG_project/
├── app_qa.py            # 问答页面（Streamlit 入口）
├── app_file_upload.py   # 知识库上传页面（Streamlit 入口）
├── rag.py               # RAG 服务：检索 + Prompt + 模型 + 会话历史
├── knowledge_base.py    # 知识库写入：MD5 去重、文本切分、向量化入库
├── vector_stores.py     # 向量检索服务：初始化 Chroma 并产出 retriever
├── file_history_store.py# 基于 JSON 文件的对话历史存储
├── config_data.py       # 全局配置（模型、切分参数、检索数量等）
├── data/                # 示例知识库文本（尺码推荐 / 洗涤养护 / 颜色选择）
├── requirements.txt     # 依赖清单
└── .env.example         # 环境变量示例
```

> `chroma_db/`、`chat_history/`、`md5.text` 均为运行时生成的产物，已在 `.gitignore` 中忽略。

## 快速开始

### 1. 准备环境

```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

### 2. 配置 API Key

代码从环境变量读取密钥，**仓库中不保存任何密钥**。

PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

Linux / macOS：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

### 3. 构建知识库

先启动上传页面，把 `data/` 下的三个 TXT 依次上传：

```bash
streamlit run app_file_upload.py
```

每个文件会提示「内容已经成功载入向量库」或「内容已经存在知识库」。

### 4. 启动问答

```bash
streamlit run app_qa.py
```

示例提问：`我的体重180，尺码推荐`、`这件衣服怎么洗`。

## 配置说明

主要参数集中在 `config_data.py`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `embedding_model` | `text-embedding-v4` | 嵌入模型 |
| `chat_model_name` | `qwen3.8-max` | 对话模型 |
| `chunk_size` / `chunk_overlap` | 1000 / 100 | 文本切分长度与重叠 |
| `similarity_threshold` | 2 | 检索返回的文档数量（k） |
| `persist_directory` | `./chroma_db` | 向量库目录 |

## 说明

- 若没有可用的 DashScope Key，`RagService` 初始化时会因嵌入模型鉴权失败而报错。
- `chroma_db/` 未纳入版本管理：换机器后需重新执行第 3 步构建知识库；如不想重建，也可把整个 `chroma_db/` 目录拷贝过去。
- 示例数据仅用于演示，实际业务请替换 `data/` 下的文档。
