import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

MODELO = "gemini-2.5-flash"

st.set_page_config(page_title="Chat de Busca Literal", layout="wide")

# -----------------------------
# CSS DO CHAT CUSTOMIZADO
# -----------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 900px;
    margin: 0 auto;
    padding-bottom: 40px;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}

.bot-message {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}

.bubble-user {
    background-color: #DCF8C6;
    color: #000000;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    word-wrap: break-word;
}

.bubble-bot {
    background-color: #F1F0F0;
    color: #000000;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    word-wrap: break-word;
}

.avatar {
    font-size: 22px;
    margin: 0 8px;
    display: flex;
    align-items: flex-end;
}

.top-box {
    max-width: 900px;
    margin: 0 auto 20px auto;
    padding: 16px;
    border: 1px solid #DDD;
    border-radius: 14px;
    background-color: #FAFAFA;
}

.arquivo-item {
    padding: 6px 10px;
    border-radius: 8px;
    background-color: #F5F5F5;
    margin-bottom: 6px;
    font-size: 14px;
}

.fonte-box {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 12px;
    margin-top: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 42px;
}

div[data-testid="stFileUploader"] {
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# FUNÇÕES
# -----------------------------
def carregar_documentos(files):
    """Carrega PDFs enviados pelo usuário."""
    documentos = []

    if not files:
        return documentos

    for arquivo in files:
        caminho_temp = None
        try:
            if not arquivo.name.lower().endswith(".pdf"):
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(arquivo.getvalue())
                caminho_temp = tmp.name

            loader = PyPDFLoader(caminho_temp)
            docs = loader.load()

            for doc in docs:
                doc.metadata["documento"] = arquivo.name

            documentos.extend(docs)

        except Exception as e:
            st.error(f"Erro ao carregar {arquivo.name}: {e}")

        finally:
            if caminho_temp and os.path.exists(caminho_temp):
                os.remove(caminho_temp)

    return documentos


def busca_literal(termo, documentos):
    """Busca literal exata nos documentos."""
    termo = termo.lower().strip()
    resultados = []

    for doc in documentos:
        texto = doc.page_content.lower()

        if termo in texto:
            doc.metadata["ocorrencias"] = texto.count(termo)
            resultados.append(doc)

    resultados.sort(
        key=lambda d: d.metadata.get("ocorrencias", 0),
        reverse=True
    )

    return resultados


def extrair_termo(pergunta):
    """Remove prefixos comuns para obter o termo da busca."""
    termo = pergunta.lower().strip()

    prefixos = [
        "procure a palavra",
        "buscar a palavra",
        "busque a palavra",
        "encontre a palavra",
        "procure por",
        "busque por",
        "buscar por",
    ]

    for prefixo in prefixos:
        if termo.startswith(prefixo):
            termo = termo.replace(prefixo, "", 1).strip()
            break

    termo = termo.strip("\"' ")
    return termo


def pergunta_resposta_literal(pergunta, documentos):
    """
    Pipeline de busca literal:
    - extrai o termo da pergunta
    - faz busca textual exata
    - monta contexto
    - gera resposta
    """
    termo = extrair_termo(pergunta)

    resultados = busca_literal(termo, documentos)
    fontes = resultados[:2]

    if not fontes:
        return f"Não encontrei a palavra '{termo}' nos documentos enviados.", []

    contexto_texto = "\n\n".join([doc.page_content for doc in fontes])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Você é um assistente corporativo.
Responda apenas com base no contexto fornecido.
A busca foi literal. Foque apenas nas ocorrências exatas do termo pesquisado.
Se a resposta não estiver no contexto, diga que não encontrou a informação.
"""),
        ("human", "Termo pesquisado: {termo}\n\nContexto:\n{contexto_texto}\n\nPergunta:\n{pergunta}")
    ])

    modelo = ChatGoogleGenerativeAI(
        model=MODELO,
        temperature=0,
        google_api_key=key
    )

    mensagem = prompt.format_messages(
        termo=termo,
        contexto_texto=contexto_texto,
        pergunta=pergunta
    )

    resposta = modelo.invoke(mensagem)

    return resposta.content, fontes


def render_user_message(texto):
    st.markdown(
        f"""
        <div class="user-message">
            <div class="bubble-user">{texto}</div>
            <div class="avatar">🧑</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_bot_message(texto):
    st.markdown(
        f"""
        <div class="bot-message">
            <div class="avatar">🤖</div>
            <div class="bubble-bot">{texto}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# ESTADO
# -----------------------------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "documentos" not in st.session_state:
    st.session_state.documentos = []

if "nomes_arquivos" not in st.session_state:
    st.session_state.nomes_arquivos = []


# -----------------------------
# CABEÇALHO
# -----------------------------
st.title("💬 Chat de Busca Literal em PDFs")

st.markdown('<div class="top-box">', unsafe_allow_html=True)

arquivos = st.file_uploader(
    "Faça upload dos documentos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Carregar PDFs"):
        if not arquivos:
            st.warning("Envie ao menos um PDF.")
        else:
            with st.spinner("Carregando documentos..."):
                documentos = carregar_documentos(arquivos)

                if documentos:
                    st.session_state.documentos = documentos
                    st.session_state.nomes_arquivos = [a.name for a in arquivos]
                    st.success("Documentos carregados com sucesso.")
                else:
                    st.warning("Nenhum conteúdo foi carregado.")

with col2:
    if st.button("Limpar conversa"):
        st.session_state.mensagens = []
        st.rerun()


# -----------------------------
# HISTÓRICO DO CHAT
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.mensagens:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        render_bot_message(msg["content"])

        if msg.get("fontes"):
            with st.expander("Fontes utilizadas"):
                for i, doc in enumerate(msg["fontes"], start=1):
                    pagina = doc.metadata.get("page")
                    pagina_exibicao = pagina + 1 if isinstance(pagina, int) else "N/A"

                    st.markdown('<div class="fonte-box">', unsafe_allow_html=True)
                    st.markdown(f"**Trecho {i}**")
                    st.write(f"**Documento:** {doc.metadata.get('documento', 'N/A')}")
                    st.write(f"**Página:** {pagina_exibicao}")
                    st.write(f"**Ocorrências do termo:** {doc.metadata.get('ocorrencias', 0)}")
                    st.write(doc.page_content)
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# INPUT DO CHAT
# -----------------------------
with st.form("form_chat", clear_on_submit=True):
    pergunta = st.text_input(
        "Mensagem",
        placeholder="Ex.: Procure a palavra criptografia",
        label_visibility="collapsed"
    )
    enviar = st.form_submit_button("Enviar")


if enviar and pergunta:
    # Salva pergunta
    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta
    })

    # Sem documentos carregados
    if not st.session_state.documentos:
        resposta_texto = "Nenhum PDF foi carregado. Envie os documentos antes de pesquisar."
        fontes = []
    else:
        with st.spinner("Consultando documentos..."):
            resposta_texto, fontes = pergunta_resposta_literal(
                pergunta,
                st.session_state.documentos
            )

    # Streaming visual da resposta
    st.session_state.mensagens.append({
        "role": "assistant",
        "content": "",
        "fontes": fontes
    })

    st.rerun()


# -----------------------------
# STREAMING CONTROLADO
# -----------------------------
if st.session_state.mensagens:
    ultima_msg = st.session_state.mensagens[-1]

    if (
        ultima_msg["role"] == "assistant"
        and ultima_msg["content"] == ""
    ):
        pergunta_usuario = st.session_state.mensagens[-2]["content"]

        if not st.session_state.documentos:
            resposta_texto = "Nenhum PDF foi carregado. Envie os documentos antes de pesquisar."
            fontes = []
        else:
            resposta_texto, fontes = pergunta_resposta_literal(
                pergunta_usuario,
                st.session_state.documentos
            )

        st.session_state.mensagens[-1]["fontes"] = fontes

        placeholder = st.empty()
        texto_parcial = ""

        for caractere in resposta_texto:
            texto_parcial += caractere
            placeholder.markdown(
                f"""
                <div class="bot-message">
                    <div class="avatar">🤖</div>
                    <div class="bubble-bot">{texto_parcial}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.01)

        st.session_state.mensagens[-1]["content"] = resposta_texto
        st.rerun()
