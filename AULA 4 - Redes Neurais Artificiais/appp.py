# ============================================================
# IA DO SORVETE 🍦
# ============================================================
# Objetivo:
# Prever a quantidade de sorvetes vendidos
# com base na temperatura.
#
# Modelo:
# LinearRegression (Scikit-Learn)
#
# Bibliotecas necessárias:
# pip install pandas scikit-learn streamlit plotly
#
# Executar:
# streamlit run app.py
# ============================================================

# =========================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =========================

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression

# =========================
# BASE DE DADOS
# =========================

sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# =========================
# VARIÁVEIS
# =========================

X = sorvete[['temperatura']]
y = sorvete['vendas']

# =========================
# MODELO
# =========================

modelo = LinearRegression()
modelo.fit(X, y)

# =========================
# INTERFACE STREAMLIT
# =========================

st.title("🍦 IA do Sorvete")

st.write("""
Modelo de Machine Learning para prever vendas de sorvete
com base na temperatura ambiente.
""")

# =========================
# DADOS
# =========================

st.subheader("📊 Dados Históricos")
st.dataframe(sorvete)

# =========================
# INPUT DO USUÁRIO
# =========================

temp = st.slider(
    "Temperatura (°C):",
    min_value=10,
    max_value=45,
    value=25
)

# =========================
# PREVISÃO
# =========================

previsao = modelo.predict([[temp]])

st.subheader("📈 Previsão de Vendas")

st.success(f"Com {temp}°C, previsão de vendas: {previsao[0]:.0f} sorvetes")

# =========================
# GRÁFICO
# =========================

st.subheader("📉 Relação Temperatura x Vendas")

sorvete['previsao'] = modelo.predict(X)

fig = px.scatter(
    sorvete,
    x='temperatura',
    y='vendas',
    title='Temperatura vs Vendas de Sorvete',
    labels={
        'temperatura': 'Temperatura (°C)',
        'vendas': 'Vendas'
    }
)

fig.add_scatter(
    x=sorvete['temperatura'],
    y=sorvete['previsao'],
    mode='lines',
    name='Regressão Linear'
)

fig.add_scatter(
    x=[temp],
    y=[previsao[0]],
    mode='markers',
    name='Previsão'
)

st.plotly_chart(fig)

# =========================
# MODELO INFO
# =========================

st.subheader("🧠 Modelo")

coef = modelo.coef_[0]
intercept = modelo.intercept_

st.write(f"Coeficiente: {coef:.2f}")
st.write(f"Intercepto: {intercept:.2f}")

st.write("### Fórmula da Regressão Linear")

st.latex("y = ax + b")

st.write(
    f"vendas = ({coef:.2f} × temperatura) + ({intercept:.2f})"
)


