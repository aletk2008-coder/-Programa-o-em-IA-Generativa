# =========================================================
# PREVISÃO DE NOTAS COM SCIKIT-LEARN + STREAMLIT
# =========================================================
# Objetivo:
# Prever a nota de um aluno com base nas horas de estudo
# utilizando Regressão Linear.
#
# Bibliotecas necessárias:
# pip install pandas scikit-learn matplotlib streamlit
#
# Para executar:
# streamlit run app.py
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression

# =========================================================
# DADOS
# =========================================================

dados = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# =========================================================
# VARIÁVEIS
# =========================================================

# X = variável independente (horas)
X = dados[['horas']]

# y = variável dependente (notas)
y = dados['notas']

# =========================================================
# TREINAMENTO DO MODELO
# =========================================================

modelo = LinearRegression()

# Treina o modelo
modelo.fit(X, y)

# =========================================================
# INTERFACE STREAMLIT
# =========================================================

st.title("📘 Previsão de Notas com Scikit-Learn")

st.write("""
Este projeto utiliza **Machine Learning** com a biblioteca
Scikit-Learn para prever notas com base nas horas de estudo.
""")

# Mostrar tabela
st.subheader("📊 Base de Dados")
st.dataframe(dados)

# Entrada do usuário
horas_estudo = st.slider(
    "Selecione as horas de estudo:",
    min_value=1,
    max_value=15,
    value=5
)

# =========================================================
# PREVISÃO
# =========================================================

previsao = modelo.predict([[horas_estudo]])

st.subheader("🎯 Resultado da Previsão")

st.write(
    f"Para {horas_estudo} horas de estudo, "
    f"a nota prevista é: **{previsao[0]:.2f}**"
)

# =========================================================
# GRÁFICO
# =========================================================

st.subheader("📈 Representação Gráfica")

fig, ax = plt.subplots()

# Pontos reais
ax.scatter(
    dados['horas'],
    dados['notas']
)

# Linha de regressão
ax.plot(
    dados['horas'],
    modelo.predict(X)
)

# Novo ponto previsto
ax.scatter(
    horas_estudo,
    previsao[0],
    s=120
)

# Títulos
ax.set_xlabel("Horas de Estudo")
ax.set_ylabel("Notas")
ax.set_title("Regressão Linear")

# Exibir gráfico
st.pyplot(fig)

# =========================================================
# INFORMAÇÕES DO MODELO
# =========================================================

st.subheader("🧠 Informações do Modelo")

st.write(f"Coeficiente Angular: {modelo.coef_[0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_:.2f}")

# Fórmula da reta
st.write("### Fórmula da Regressão Linear")


::contentReference[oaicite:0]{index=0}


st.write(
    f"Modelo encontrado: "
    f"nota = ({modelo.coef_[0]:.2f} × horas) + "
    f"{modelo.intercept_:.2f}"
)