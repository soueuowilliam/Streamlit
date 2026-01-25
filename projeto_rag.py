# rag_core.py
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def carregar_documentos(caminho):
    documentos = []

    if os.path.isdir(caminho):
        for arquivo in os.listdir(caminho):
            arquivo_norm = arquivo.strip().lower()
            caminho_arquivo = os.path.join(caminho, arquivo)

            if not os.path.isfile(caminho_arquivo):
                continue

            try:
                if arquivo_norm.endswith('.pdf'):
                    documentos.extend(PyPDFLoader(caminho_arquivo).load())
                elif arquivo_norm.endswith('.doc'):
                    documentos.extend(Docx2txtLoader(caminho_arquivo).load())
                else:
                    continue
                    
            except Exception as e:
                print(f'Falha ao ler {arquivo}: {e}')

    if not documentos:
        raise RuntimeError('Nenhum documento encontrado.')

    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
    )

    splitter = RecursiveCharacterTextSplitter( chunk_size = 500, chunk_overlap = 100 )
    chunks = splitter.split_documents(documentos)

    retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs = {"k": 3})
    
    return retriever

def formatar_contexto(docs):
    textos = []
    for doc in docs:
        arquivo = os.path.basename(doc.metadata.get("source", "N/A"))
        pagina = doc.metadata.get("page")
        pagina_txt = f"Página {pagina + 1}" if pagina is not None else "Página não aplicável"
        textos.append(f"📄 {arquivo} | {pagina_txt}\n{doc.page_content}")
    return "\n\n".join(textos)

_chain = None

def inicializacao(pasta:str):
    global _chain
    retriever = carregar_documentos(pasta)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    Você é um assistente de busca inteligente com a função de fazer a leitura dos documentos.
    Você deve localizar a ocorrência literal da PALAVRA_ALVO no contexto fornecido.
    
    REGRAS:
    - A busca deve ser EXATA e CASE-INSENSITIVE.
    - Não considere sinônimos, variações morfológicas, ou correspondências parciais que não incluam a PALAVRA_ALVO literal.
    - Use SOMENTE o contexto fornecido (não use conhecimento externo).
    - Se não houver ocorrências, responda exatamente:
      " A palavra 'palavra' não encontrada nos documentos fornecidos."
    - Para cada ocorrência, retorne no formato abaixo (NÃO altere o formato):
     
        Resposta:
        Ocorrências encontradas:
        - Documento: <nome_do_documento>
        > Página: <número_da_página>
        > Trecho: "<frase ou parágrafo onde a palavra aparece>"
    
    (Repita para cada ocorrência encontrada)
    
    Contexto:
    {contexto}
        """
        ),
        ("human", "{pergunta}")
    ])

    
    modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=api_key)
    _chain = (
        {
        'contexto': RunnablePassthrough() | retriever | formatar_contexto ,
        'pergunta': RunnablePassthrough()
        } | prompt | modelo | StrOutputParser()
    )


def responder(pergunta: str) -> str: 
    if _chain is None:
        raise RuntimeError('Modelo não inicializado')
    return _chain.invoke(pergunta)




















