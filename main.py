import streamlit as st  
import requests 
from datetime import datetime  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import StandardScaler 


st.set_page_config(page_title="Dashboard Bitcoin", layout="wide")  


# Título principal da dashboard
st.title("Análise do Preço do Bitcoin")

# Função para buscar os dados da API
@st.cache_data 
def fetch_data(days=30):
    """Busca dados do Bitcoin nos últimos 'days' dias."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "brl", "days": str(days)}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = [item[1] for item in data["prices"]]
        timestamps = [datetime.fromtimestamp(item[0] / 1000) for item in data["prices"]]
        return timestamps, prices
    else:
        st.error("Erro ao buscar dados!")
        return [], []

# Controle deslizante para o usuário escolher o intervalo de dias
dias = st.slider(
    "Selecione o intervalo de dias", 
    min_value=10, 
    max_value=365, 
    value=30, 
    step=1
)


# Busca os dados com base no valor selecionado no slider
timestamps, prices = fetch_data(dias)

# Verifica se os dados foram carregados
if len(prices) == 0:
    st.warning("Nenhum dado encontrado para o intervalo selecionado.")
else:
    st.success("Dados carregados com sucesso!")  

    # Calcula estatísticas dos preços
    precos_array = np.array(prices)
    media = np.mean(precos_array)
    minimo = np.min(precos_array)
    maximo = np.max(precos_array)
    desvio = np.std(precos_array)

    # Mostra estatísticas na dashboard
    st.subheader("Estatísticas dos últimos dias")
    st.metric("Preço Médio", f"R${media:,.2f}")
    st.metric("Menor Preço", f"R${minimo:,.2f}")
    st.metric("Maior Preço", f"R${maximo:,.2f}")
    st.metric("Desvio Padrão", f"R${desvio:,.2f}")

    # Gráfico da variação dos preços
    fig, ax = plt.subplots(figsize=(10, 5))  
    ax.plot(timestamps, prices, color="blue", label="Preço do Bitcoin")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (BRL)")
    ax.set_title("Variação do Bitcoin nos Últimos Dias")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)  
    st.pyplot(fig)  

    # Regressão Linear para previsão
    dias_recentes = np.array(range(len(prices))).reshape(-1, 1)
    scaler = StandardScaler()
    dias_recentes_scaled = scaler.fit_transform(dias_recentes)
    model = LinearRegression()
    model.fit(dias_recentes_scaled, prices)

    # Previsão do próximo dia
    next_day_scaled = scaler.transform([[len(prices)]])
    preco_previsto = model.predict(next_day_scaled)

    st.subheader("Previsão do próximo preço")
    st.metric("Próximo Preço Estimado", f"R${preco_previsto[0]:,.2f}")
  