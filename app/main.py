import os
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Cria um callable para a entrada do usuário
class CustomRunnable:
    def __init__(self, content):
        self.content = content

    def __call__(self):
        return self.content


# Função principal
def main():
    # Carregar variáveis de ambiente
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    model_name = 'llama3-8b-8192'

    # Inicializar o objeto de chat do Groq
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name
    )

    # Inicializar LangGraph para gerenciar contexto de conversa
    state_schema = (
        ("input", ()),
        ("output", ()),
    )
    knowledge_graph = StateGraph(state_schema=state_schema)

    st.title("Groq Chatbot com LangGraph e FAISS")
    st.subheader("Me faça qualquer pergunta!")
    # st.write("Vamos começar nossa conversa!")

    # Prompt do chatbot
    system_prompt = 'Você é um chatbot amigável e conversacional.'
    
    # Número de mensagens anteriores que o chatbot lembrará durante a conversa
    conversational_memory_length = 5

    # Inicializar o estado da sessão para o histórico de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Criar o buffer de memória para armazenar o histórico da conversa
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Inicializar o modelo de embeddings (apenas tokenizer)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)

    # Inicializar a base de dados FAISS
    dimension = 384  # Dimensão dos embeddings para o modelo
    faiss_index = faiss.IndexFlatL2(dimension)

    def get_embeddings(text):
        # Tokeniza o texto
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Use o modelo para obter os embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            # Pega os embeddings da última camada
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Verifica se o tamanho está correto
        if embeddings.shape[0] != dimension:
            raise ValueError(f"Dimensão incorreta: esperado {dimension}, mas obtido {embeddings.shape[0]}.")

        return embeddings

    input_text = st.text_area("Faça uma pergunta:")

    if input_text:
        # Criar um callable para a entrada do usuário
        user_input_runnable = CustomRunnable(input_text)

        # Adicionar a entrada do usuário ao grafo de conhecimento
        knowledge_graph.add_node("User Input", user_input_runnable)

        # Criar um modelo de prompt de conversa utilizando componentes do LangChain
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        # Atualizar o histórico de chat
        st.session_state.chat_history.append(f"Human: {input_text}")

        # Criar uma cadeia de conversa (LLMChain) sem passar diretamente o memory
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False
        )

        # Gerar resposta do chatbot com o histórico atualizado
        response = conversation.predict(human_input=input_text, chat_history=st.session_state.chat_history)

        # Adicionar a resposta do chatbot ao histórico de chat
        st.session_state.chat_history.append(f"Chatbot: {response}")

        # Criar um callable para a resposta do chatbot
        chatbot_response_runnable = CustomRunnable(response)

        # Adicionar a resposta do chatbot ao grafo de conhecimento
        knowledge_graph.add_node("Chatbot Response", chatbot_response_runnable)

        # Gerar o embedding da resposta para armazenar no FAISS
        response_embedding = get_embeddings(response)

        # Redimensionar e adicionar ao índice FAISS
        response_embedding = response_embedding.reshape(1, -1)
        faiss_index.add(response_embedding)

        # Implementar validação para garantir que a resposta é relevante
        def validate_response(response):
            if "falso" in response.lower() or "não verdadeiro" in response.lower():
                return False
            return True

        if validate_response(response):
            st.write(f"Resposta validada: {response}")
        else:
            st.write("Resposta não validada, buscando uma melhor...")

    # Exibir todo o histórico de chat
    for message in st.session_state.chat_history:
        st.write(message)

    # Função para buscar respostas semelhantes usando FAISS
    def search_similar_responses(query):
        query_embedding = get_embeddings(query)
        _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=1)
        return indices

    # Busca por respostas semelhantes
    if input_text:
        similar_responses_idx = search_similar_responses(input_text)
        if similar_responses_idx[0][0] != -1:
            st.write("Resposta semelhante encontrada: ", st.session_state.chat_history[similar_responses_idx[0][0]])
        else:
            st.write("Nenhuma resposta semelhante encontrada.")


if __name__ == "__main__":
    main()