# ============================================================
# 🥷 DETECTOR DE APROVAÇÃO NINJA
# ============================================================
# Objetivo:
# Classificar alunos em:
# - Aprovado (1)
# - Reprovado (0)
#
# Modelo:
# LogisticRegression (Scikit-Learn)
#
# Bibliotecas necessárias:
# pip install pandas scikit-learn streamlit plotly
#
# Executar:
# streamlit run app.py
# ============================================================

# =========================
# IMPORTAÇÃO
# =========================

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression

# =========================
# BASE DE DADOS
# =========================

alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]  # 1 = aprovado | 0 = reprovado
})

# =========================
# VARIÁVEIS
# =========================

X = alunos[['faltas']]
y = alunos['resultado']

# =========================
# MODELO
# =========================

modelo = LogisticRegression()
modelo.fit(X, y)

# =========================
# STREAMLIT
# =========================

st.title("🥷 Detector de Aprovação Ninja")

st.write("""
Sistema de Machine Learning que classifica alunos como:
- 🟢 Aprovado
- 🔴 Reprovado

com base no número de faltas.
""")

# =========================
# DADOS
# =========================

st.subheader("📊 Dados de Treinamento")
st.dataframe(alunos)

# =========================
# INPUT
# =========================

faltas = st.slider(
    "Número de faltas:",
    min_value=0,
    max_value=15,
    value=3
)

# =========================
# PREVISÃO
# =========================

pred = modelo.predict([[faltas]])[0]
prob = modelo.predict_proba([[faltas]])[0]

# =========================
# RESULTADO
# =========================

st.subheader("🎯 Resultado da Classificação")

if pred == 1:
    st.success("🟢 Aprovado Ninja!")
else:
    st.error("🔴 Reprovado Ninja!")

st.write(f"Probabilidade de aprovação: {prob[1]:.2f}")

# =========================
# GRÁFICO
# =========================

st.subheader("📈 Visualização")

alunos['prob_aprovacao'] = modelo.predict_proba(X)[:, 1]

fig = px.scatter(
    alunos,
    x='faltas',
    y='resultado',
    title='Faltas vs Aprovação Ninja',
    labels={
        'faltas': 'Faltas',
        'resultado': 'Aprovação (0/1)'
    }
)

fig.add_scatter(
    x=alunos['faltas'],
    y=alunos['prob_aprovacao'],
    mode='lines',
    name='Probabilidade (Logistic Regression)'
)

fig.add_scatter(
    x=[faltas],
    y=[prob[1]],
    mode='markers',
    name='Previsão Atual'
)

st.plotly_chart(fig)

# =========================
# MODELO INFO
# =========================

st.subheader("🧠 Modelo")

st.write(f"Coeficiente: {modelo.coef_[0][0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_[0]:.2f}")

st.write("### Função logística")

st.latex(r"P(y) = \frac{1}{1 + e^{-(ax + b)}}")