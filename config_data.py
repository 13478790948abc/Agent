
md5_path = "./md5.text"
collection_name = "rag"
persist_directory = "./chroma_db"

# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "。", "!", "?", "？", " ", ""]
max_spliter_char_number=1000            # 文本分割的阈值

# 相似度检索阈值, 检索返回匹配的文档数量
similarity_threshold = 2


embedding_model = "text-embedding-v4"
chat_model_name = "qwen3.8-max"

# ChatOpenAI 走 DashScope 的 OpenAI 兼容模式端点，qwen3.8 系列模型必须用这个端点
chat_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}