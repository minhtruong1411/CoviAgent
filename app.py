import streamlit as st
import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import Settings

# Thêm thư mục gốc vào path để import
sys.path.append(str(Path(__file__).parent))

from coviagent.core.config import AppConfig
from coviagent.core.factories import create_llm, create_embedding
from coviagent.retrieval.vector_retriever import VectorRetriever
from coviagent.pipeline.vector_rag import VectorRAGPipeline

st.set_page_config(page_title="CoviAgent Chatbot", page_icon="🩺", layout="centered")

@st.cache_resource
def init_pipeline(config_path, pipeline_type="vector"):
    load_dotenv()
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    import os
    actual_api_key = config.llm.api_key
    if not actual_api_key and getattr(config.llm, "api_key_env_var", None):
        actual_api_key = os.environ.get(config.llm.api_key_env_var)

    # 1. Knowledge Graph RAG
    if pipeline_type == "kg":
        from coviagent.pipeline.kg_rag import KGRAGPipeline
        return KGRAGPipeline(model_name=config.llm.model_name, api_key=actual_api_key)
        
    embed_model = create_embedding(config.embedding)
    llm = create_llm(config.llm)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    client = QdrantClient(url=config.qdrant_url)
    retriever = VectorRetriever.from_qdrant_client(
        client=client, 
        config=config.retriever,
        embed_model=embed_model,
        reranker_config=config.reranker
    )
    
    # 2. GraphRAG
    if pipeline_type == "graphrag":
        from coviagent.pipeline.graph_rag import GraphRAGPipeline
        neo4j_cfg = config.neo4j
        return GraphRAGPipeline(
            retriever=retriever,
            llm=llm,
            neo4j_uri=neo4j_cfg.uri if neo4j_cfg else None,
            neo4j_username=neo4j_cfg.username if neo4j_cfg else None,
            neo4j_password=neo4j_cfg.password if neo4j_cfg else None,
            neo4j_database=neo4j_cfg.database if neo4j_cfg else None,
        )

    # 3. Vector RAG (Default)
    return VectorRAGPipeline(retriever=retriever, llm=llm)

st.title("🩺 CoviAgent Medical Chatbot")
st.markdown("Hệ thống hỏi đáp y tế chuyên nghiệp sử dụng công nghệ RAG tiên tiến.")

# Sidebar để tuỳ chỉnh cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    
    pipeline_type = st.selectbox(
        "🧠 Chọn phương pháp RAG:",
        options=["vector", "kg", "graphrag"],
        format_func=lambda x: {
            "vector": "Vector RAG (Cơ bản)", 
            "kg": "Knowledge Graph (Logic)", 
            "graphrag": "GraphRAG (Kết hợp đỉnh cao)"
        }[x],
        index=2 # Mặc định chọn GraphRAG
    )
    
    config_file = st.selectbox(
        "⚙️ Chọn file cấu hình LLM:",
        ["configs/judge_benchmark.yaml", "configs/openrouter_hf.yaml", "configs/default_vector_rag.yaml", "configs/groq_hf_embed.yaml"]
    )
    
    if st.button("🚀 Khởi động lại hệ thống"):
        st.cache_resource.clear()
        st.rerun()

try:
    pipeline = init_pipeline(config_file, pipeline_type)
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.info("Hãy đảm bảo Qdrant và Neo4j đang chạy (docker-compose up -d)")
    st.stop()

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input của người dùng
if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về COVID-19..."):
    # Hiển thị câu hỏi
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Hiển thị câu trả lời
    with st.chat_message("assistant"):
        # Tạo hiệu ứng suy luận (Reasoning) để làm nổi bật GraphRAG
        with st.status("🧠 Đang phân tích truy vấn...", expanded=True) as status:
            st.write("🔍 **Bước 1:** Tìm kiếm ngữ nghĩa (Vector Search)...")
            
            # Nếu là GraphRAG thì cập nhật trạng thái thêm
            if pipeline_type == "graphrag":
                st.write("🕸️ **Bước 2:** Duyệt Đồ thị Tri thức (Knowledge Graph Expansion)...")
            elif pipeline_type == "kg":
                st.write("🕸️ **Bước 1:** Trích xuất Thực thể & Duyệt Knowledge Graph...")
                
            st.write("✨ **Bước cuối:** Đang dùng LLM tổng hợp câu trả lời...")
            
            # Gọi hàm query
            try:
                response = pipeline.query(prompt)
                answer = response.get('answer', "Không có câu trả lời.")
                status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Lỗi xử lý", state="error", expanded=False)
                st.error(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
                st.stop()
            
        # In ra câu trả lời chính
        st.markdown(answer)
        
        # Tách bạch rõ các nguồn tài liệu tham khảo
        if response.get('contexts'):
            has_graph = 'graph_contexts' in response
            has_vector = 'vector_contexts' in response
            
            if has_graph or has_vector:
                # Hiển thị cho GraphRAG / KG
                total = len(response['contexts'])
                with st.expander(f"📚 Cấu trúc Tri thức Truy xuất ({total} nguồn)", expanded=True):
                    
                    if has_vector and response['vector_contexts']:
                        st.markdown("#### 🎯 Nguồn từ Vector Search (Ngữ nghĩa)")
                        for i, ctx in enumerate(response['vector_contexts']):
                            st.info(f"**Đoạn {i+1}:** {ctx}")
                            
                    if has_graph:
                        st.markdown("#### 🕸️ Nguồn mở rộng từ Graph (Logic/Lân cận)")
                        if response['graph_contexts']:
                            for i, ctx in enumerate(response['graph_contexts']):
                                st.success(f"**Mở rộng {i+1}:** {ctx}")
                        else:
                            st.warning("⚠️ Không tìm thấy liên kết đồ thị bổ sung nào cho truy vấn này (Hoặc Neo4j chưa được khởi động đúng cách).")
            else:
                # Hiển thị mặc định cho Vector RAG thường
                with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(response['contexts'])} đoạn)"):
                    for i, ctx in enumerate(response['contexts']):
                        st.markdown(f"**Đoạn {i+1}:**\n{ctx}")
                        st.divider()
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
