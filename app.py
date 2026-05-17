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
def init_pipeline(config_path):
    load_dotenv()
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    embed_model = create_embedding(config.embedding)
    llm = create_llm(config.llm)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    client = QdrantClient(url=config.qdrant_url)
    retriever = VectorRetriever.from_qdrant_client(
        client=client, 
        config=config.retriever,
        embed_model=embed_model
    )
    return VectorRAGPipeline(retriever=retriever, llm=llm)

st.title("🩺 CoviAgent Medical Chatbot")
st.markdown("Hệ thống hỏi đáp y tế chuyên nghiệp dựa trên công nghệ RAG.")

# Sidebar để tuỳ chỉnh cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình")
    config_file = st.selectbox(
        "Chọn file cấu hình:",
        ["configs/openrouter_hf.yaml", "configs/default_vector_rag.yaml", "configs/groq_hf_embed.yaml"]
    )
    if st.button("Tải lại cấu hình"):
        st.cache_resource.clear()
        st.rerun()

try:
    pipeline = init_pipeline(config_file)
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.info("Hãy đảm bảo Qdrant đang chạy (docker-compose up -d)")
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
        with st.spinner("Đang tổng hợp thông tin..."):
            try:
                response = pipeline.query(prompt)
                answer = response['answer']
                
                st.markdown(answer)
                
                # Hiển thị context tham khảo trong expander
                if response.get('contexts'):
                    with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(response['contexts'])} đoạn)"):
                        for i, ctx in enumerate(response['contexts']):
                            st.markdown(f"**Đoạn {i+1}:**\n{ctx}")
                            st.divider()
                            
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
