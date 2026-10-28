# RAG de vendas de carros

Projeto independente para o desafio da DIO: conectar dados próprios a um chatbot RAG local e persistir os vetores em ChromaDB.

O projeto anterior foi preservado no arquivo `../portfolio-rag-vendas-atual.zip`. Esta pasta não altera nem substitui aquele chatbot.

## O que o pipeline faz

1. Lê os três CSVs relacionados em `data/`.
2. Junta cada venda ao veículo e à região do cliente para formar documentos de contexto.
3. Converte os documentos em embeddings com FastEmbed.
4. Armazena e reutiliza os vetores em `storage/`, usando ChromaDB persistente.
5. Recupera os registros mais relevantes para a pergunta e envia apenas esse contexto ao modelo configurado no OpenRouter.

O modelo não é treinado com a planilha. Para uma pergunta, o RAG busca registros relacionados e os inclui no contexto enviado ao LLM.

## Os três CSVs

- `data/vendas.csv`: identificador da venda, veículo, cliente, data e preço.
- `data/veiculos.csv`: marca e modelo associados a um identificador de veículo.
- `data/clientes.csv`: país e região associados a um identificador anônimo de cliente.

Os nomes de clientes da planilha original não são exportados. Os identificadores de venda são gerados para serem únicos porque a coluna `ID` da origem contém uma duplicidade. A planilha original não é copiada para este projeto.

Para gerar novamente os CSVs a partir de uma planilha com a mesma estrutura:

```powershell
python -m pip install -r requirements.txt
python scripts/export_data.py "C:\caminho\para\base_dados.xlsx"
```

O script espera uma aba chamada `base` com as colunas ID, MARCA, MODELO, PREÇO, PAÍS, REGIÃO, NOME e DATA.

## Executar com Docker

1. Copie `.env.example` para `.env` e preencha `OPENROUTER_API_KEY` com sua própria chave.
2. Na pasta `carros-rag`, execute:

```powershell
docker compose up --build
```

3. Abra <http://localhost:8001>.

O primeiro início baixa o modelo de embeddings e cria o índice; pode levar alguns minutos. Depois, o volume local `storage/` preserva o índice entre reinicializações. Para parar, use `Ctrl+C`; para executar em segundo plano, use `docker compose up --build -d`.

## Executar sem Docker

Requer Python 3.12 ou superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edite .env e configure OPENROUTER_API_KEY
chainlit run src/main.py --host 127.0.0.1 --port 8001
```

## Testes e perguntas para experimentar

```powershell
python -m unittest discover -s tests -v
```

Experimente perguntar por uma marca/modelo, por uma faixa de preço encontrada nos registros, ou sobre o país e a região de uma venda. O mecanismo atual faz recuperação semântica dos registros; não use a resposta vetorial como substituto de uma consulta exata para totais ou contagens de toda a tabela.

## Publicação segura

Antes de enviar ao GitHub, revise os dados e confirme que tem autorização para publicá-los. Os nomes de clientes foram anonimizados e a planilha de origem ficou fora do projeto. Nunca publique `.env`, chaves de API, índices locais ou ambientes virtuais.

