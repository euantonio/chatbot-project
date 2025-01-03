FROM python:3

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o conteúdo da pasta 'app' para o contêiner
COPY app/ /app/

# InstalaR as dependências
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expor a porta 8501
EXPOSE 8501

# Comando para rodar sua aplicação
CMD ["streamlit", "run", "main.py"]