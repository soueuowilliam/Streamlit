# meu_app.py
import streamlit as st
import tempfile
import os
import shutil

from projeto_rag import inicializacao, responder

st.set_page_config(
    page_title="📄 Sistema Inteligente de Busca",
    layout="wide"
)

st.title("📄 Sistema Inteligente de Busca")
st.write(
    "Carregue um ou mais documentos e faça perguntas utilizando busca semântica."
)

# =========================
# AUTENTICAÇÃO
# =========================
senha = st.text_input("Digite sua senha:", type="password")
senhaacesso = "12345"

if senha != senhaacesso:
    st.info("Por favor, insira a senha para utilizar a ferramenta.", icon="🗝️")
    st.stop()

# =========================
# UPLOAD (MÚLTIPLOS)
# =========================
uploaded_files = st.file_uploader(
    "Faça upload dos documentos (.pdf ou .docx)",
    type=("pdf", "docx"),
    accept_multiple_files = True
)

# =========================
# PROCESSAR DOCUMENTOS (1x)
# =========================
if 'pasta_docs' not in st.session_state:
    st.session_state.pasta_docs = tempfile.mkdtemp()

if uploaded_files and "rag_pronto" not in st.session_state:
    with st.spinner("Processando documentos e criando base de conhecimento..."):
        pasta = st.session_state.pasta_docs

        for file in uploaded_files:
            caminho = os.path.join(pasta, file.name)
            with open(caminho, "wb") as f:
                f.write(file.getbuffer())

        inicializacao(pasta)
        st.session_state.rag_pronto = True

    st.success(f"{len(uploaded_files)} documento(s) carregado(s) com sucesso!")

# =========================
# PERGUNTA
# =========================
with st.form('pergunta_form'):
    pergunta = st.text_input('Faça a sua pergunta')
    enviar = st.form_submit_button('Enviar')

if enviar and st.session_state.get('rag_pronto'):
    with st.spinner("Buscando resposta nos documentos..."):
        resposta = responder(pergunta)
        st.markdown(resposta)

# =========================
# RESET
# =========================
if st.button("🔄 Carregar novos documentos"):
    if "pasta_temp" in st.session_state:
        shutil.rmtree(st.session_state.pasta_temp, ignore_errors=True)

    for k in ["rag_pronto", "pasta_temp"]:
        if k in st.session_state:
            del st.session_state[k]

    st.rerun()
