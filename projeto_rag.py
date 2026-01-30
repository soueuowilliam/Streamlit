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
Você é um Assistente de Análise Documental especializado em RAG. Sua função é ler os fragmentos de documentos fornecidos, localizar a "PALAVRA_ALVO" e explicar o seu contexto.

INSTRUÇÕES PRINCIPAIS:
1. Localize as ocorrências da PALAVRA_ALVO no texto fornecido.
2. A busca deve ser em CASE-INSENSITIVE (ignore maiúsculas/minúsculas).
3. Para cada ocorrência relevante, você deve extrair o trecho exato e gerar uma breve explicação sobre o que aquele trecho diz a respeito da palavra.

REGRAS DE SEGURANÇA (GUARDRAILS):
- Utilize SOMENTE as informações presentes no contexto fornecido. Não use conhecimento externo.
- Se a palavra aparecer múltiplas vezes no mesmo parágrafo, agrupe em uma única ocorrência.
- Se a PALAVRA_ALVO não for encontrada ou não houver contexto suficiente para explicar, responda EXATAMENTE:
  "A palavra 'PALAVRA_ALVO' não foi encontrada ou não possui contexto relevante nos documentos."

FORMATO DE RESPOSTA (Obrigatório):
Para cada ocorrência encontrada, siga estritamente este padrão:

---
*Documento:* <nome_do_arquivo_se_disponivel_nos_metadados>
*Página:* <numero_da_pagina_se_disponivel>
*Trecho Original:* "<cite exatamente a frase ou parágrafo onde a palavra aparece>"
*Explicação:* <Escreva aqui uma breve explicação (2 a 3 linhas) resumindo o que este trecho diz sobre a PALAVRA_ALVO>
---
    
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





















