# Importação das bibliotecas necessárias
import requests  # Para fazer requisições HTTP à API
from datetime import datetime  # Para manipulação de datas
import matplotlib.pyplot as plt  # Para criar gráficos
import numpy as np  # Para manipulação de arrays e cálculos numéricos
from sklearn.linear_model import LinearRegression  # Para regressão linear
from sklearn.preprocessing import StandardScaler  # Para escalonamento dos dados

# URL da API do CoinGecko e parâmetros para buscar os dados do Bitcoin
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "brl", "days": "30"}  # Parâmetros: moeda (BRL) e dias (últimos 30 dias)

# Requisição HTTP para buscar os dados da API
response = requests.get(url, params=params)
if response.status_code == 200:  # Se a requisição for bem-sucedida
    data = response.json()  # Converte a resposta da API para formato JSON
    # Extrai apenas os preços do Bitcoin (item[1] é o prices na tupla) ai ele percorre apenas o item que ele colocou no começo,no caso o [item 1]
    prices = [item[1] for item in data["prices"]]  
    # Converte os timestamps para datas legíveis
    timestamps = [datetime.fromtimestamp(item[0] / 1000) for item in data["prices"]]
else:
    # Caso a requisição não seja bem-sucedida, exibe o erro e encerra o programa
    print("Erro ao buscar dados:", response.status_code)
    print(response.text)
    exit()

# Pega os últimos 30 preços dos dados retornados da API
precos_recentes = np.array(prices[-30:])  # Últimos 30 preços
# Cria uma sequência de dias de 0 a 29 (para os 30 dias)
dias_recentes = np.array(range(len(precos_recentes))).reshape(-1, 1)  # Dias sequenciais,serve para deixar em linha

# Escalonamento dos dados: ajusta os dados para uma distribuição padrão (média 0, desvio padrão 1)

scaler = StandardScaler()
dias_recentes_scaled = scaler.fit_transform(dias_recentes)  # Ajusta e aplica o escalonamento 
#Antes do escalonamento: Os valores dos dias são 0, 1, 2, ..., 29. Depois do escalonamento: Os valores são ajustados para estarem em uma escala com média 0 e desvio padrão 1. Por quê? Para que o modelo de machine learning trate todas as variáveis de forma justa, sem favorecer as que têm números maiores.

# Cria o gráfico mostrando a variação do preço do Bitcoin nos últimos 30 dias
plt.plot(timestamps[-30:], precos_recentes, color="blue", label="Preço do Bitcoin")  # Plota os dados
plt.xticks(rotation=45)  # Rotaciona as datas para ficarem legíveis
plt.title("Variação do Bitcoin nos Últimos 30 Dias")  # Título do gráfico
plt.xlabel("Data")  # Rótulo do eixo X (datas)
plt.ylabel("Preço (BRL)")  # Rótulo do eixo Y (preço em BRL)
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.legend()  # Adiciona uma legenda
 # Exibe o gráfico

# Cálculos estatísticos sobre os preços dos últimos 30 dias
average_valores = np.mean(precos_recentes)  # Média dos preços
max_valores = np.max(precos_recentes)  # Maior preço
min_valores = np.min(precos_recentes)  # Menor preço
desvio_padrao = np.std(precos_recentes)  # Desvio padrão (variação dos preços)

# Verifica se o valor do Bitcoin está aumentando ou diminuindo nos últimos dias
if precos_recentes[-1] > precos_recentes[0]:  # Compara o último preço com o primeiro
    movimento = "O valor está aumentando"
else:
    movimento = "O valor está diminuindo"

# Exibe os resultados das estatísticas
print("=====================")
print(f"Média dos preços: R${average_valores:,.2f}")
print("=====================")
print(f"Menor preço: R${min_valores:,.2f}")
print("=====================")
print(f"Maior preço: R${max_valores:,.2f}")
print("=====================")
print(f"Desvio padrão: R${desvio_padrao:,.2f}")
print("=====================")
print(f"Ultimo valor foi: R${precos_recentes[-1]:,.2f}")
print("=====================")
print(f"{movimento}")
print("=====================")


# Cria e treina o modelo de Regressão Linear usando os dados escalonados
model = LinearRegression()  # Instancia o modelo de regressão linear
model.fit(dias_recentes_scaled, precos_recentes)  # Treina o modelo com os dados de dias e preços

# Previsão do preço do próximo dia (após os últimos 30 dias)
next_day_scaled = scaler.transform([[len(precos_recentes)]])  # Escalonamento do próximo dia
preco_previsto = model.predict(next_day_scaled)  # Faz a previsão com o modelo treinado

# Exibe a previsão do próximo preço
print(f"Previsão do próximo preço: R${preco_previsto[0]:,.2f}")
print("=====================")
