"""
只读检索服务层：封装 Chroma 初始化、接收外部注入的嵌入模型、以工厂方法产出配置好 k 值的 retriever。
"""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        # 从 ./chroma_db 这个数据库目录中打开名为 "rag" 的集合，用 text-embedding-v4 把查询转成向量，
        # 再由 Chroma 在集合内做向量相似度计算，返回 top-2 文档。
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入链"""
        # 检索动作本身被推迟到 chain.invoke() 时才发生，且每次调用独立执行、结果随问题变化。
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

if __name__ == '__main__':
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体值180，尺码推荐")
    print(res)