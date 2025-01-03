
## Tecnologias Utilizadas

- [Python 3](https://www.python.org/): Linguagem base do projeto.
- [Streamlit](https://streamlit.io/): Framework para criar interface gráfica.
- [LangChain](https://langchain.readthedocs.io/): Biblioteca para criar fluxos de processamento com LLMs.
- [FAISS](https://faiss.ai/): Busca de informações baseada em similaridade de embeddings.
- [Docker](https://www.docker.com/): Ambiente de contêiner para desenvolvimento isolado e reprodutível.

## Instalação

```bash
git clone https://github.com/euantonio/chatbot-project.git
cd chatbot-project
```

Crie um arquivo `.env` na raiz do projeto com sua chave de API da Groq:
```bash
GROQ_API_KEY=SUA_CHAVE_API
```

## Criar contêiner Docker

Execute o comando:
```
docker build -t projeto-python .
```

Após a criação do contêiner, executar:
```
docker run -it projeto-python
```

## 🔍 Funcionamento do Chatbot

Fluxo Principal:

- O usuário insere uma pergunta.
- O chatbot responde utilizando o modelo configurado (Groq LLM).

Validação de Respostas:

- O projeto inclui uma função para validar se a resposta gerada é relevante.

Armazenamento em FAISS:

- As respostas são convertidas em embeddings e armazenadas em um índice FAISS para buscas futuras.

Histórico Conversacional:

- Um buffer armazena até 5 interações anteriores para manter o contexto da conversa.
