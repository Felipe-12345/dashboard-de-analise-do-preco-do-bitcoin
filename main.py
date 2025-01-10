import requests  
from datetime import datetime  
import matplotlib.pyplot as plt 
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import StandardScaler  

# URL da API do CoinGecko e parâmetros para buscar os dados do Bitcoin
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "brl", "days": "30"} 

# Requisição HTTP para buscar os dados da API
response = requests.get(url, params=params)
if response.status_code == 200: 
    data = response.json() 
    prices = [item[1] for item in data["prices"]]  
    
    timestamps = [datetime.fromtimestamp(item[0] / 1000) for item in data["prices"]]
else:
    
    print("Erro ao buscar dados:", response.status_code)
    print(response.text)
    exit()

# Pega os últimos 30 preços dos dados retornados da API
precos_recentes = np.array(prices[-30:])  

dias_recentes = np.array(range(len(precos_recentes))).reshape(-1, 1)  

# Escalonamento dos dados: ajusta os dados para uma distribuição padrão (média 0, desvio padrão 1)

scaler = StandardScaler()
dias_recentes_scaled = scaler.fit_transform(dias_recentes)   


# Cria o gráfico mostrando a variação do preço do Bitcoin nos últimos 30 dias
plt.plot(timestamps[-30:], precos_recentes, color="blue", label="Preço do Bitcoin")  
plt.xticks(rotation=45)  
plt.title("Variação do Bitcoin nos Últimos 30 Dias")  
plt.xlabel("Data")  
plt.ylabel("Preço (BRL)")  
plt.grid(True)  
plt.legend()  
 

# Cálculos estatísticos sobre os preços dos últimos 30 dias
average_valores = np.mean(precos_recentes)  
max_valores = np.max(precos_recentes)  
min_valores = np.min(precos_recentes)  
desvio_padrao = np.std(precos_recentes)  

# Verifica se o valor do Bitcoin está aumentando ou diminuindo nos últimos dias
if precos_recentes[-1] > precos_recentes[0]:  
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
model = LinearRegression()  
model.fit(dias_recentes_scaled, precos_recentes)  
# Previsão do preço do próximo dia (após os últimos 30 dias)
next_day_scaled = scaler.transform([[len(precos_recentes)]])  
preco_previsto = model.predict(next_day_scaled)  

# Exibe a previsão do próximo preço
print(f"Previsão do próximo preço: R${preco_previsto[0]:,.2f}")
print("=====================")
