import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression

gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

X = gamer[['horas_jogo']]
y = gamer['cansaco']

modelo = LinearRegression()

modelo.fit(X, y)

st.title("🎮 Detector de Sono Gamer")

st.write("""
Sistema de Inteligência Artificial utilizando
Scikit-Learn para prever o nível de cansaço
de um jogador baseado no tempo jogando.
""")

st.subheader("📊 Dados Utilizados")

st.dataframe(gamer)

horas = st.slider(
    "Horas jogando:",
    min_value=1,
    max_value=15,
    value=5
)

previsao = modelo.predict([[horas]])

st.subheader("😴 Resultado da IA")

st.success(
    f"Nível de cansaço previsto: {previsao[0]:.2f}"
)

if previsao[0] <= 3:
    st.info("🟢 Baixo cansaço")
elif previsao[0] <= 6:
    st.warning("🟡 Cansaço moderado")
else:
    st.error("🔴 Alto cansaço")

st.subheader("📈 Representação Gráfica")

gamer['previsao'] = modelo.predict(X)

grafico = px.scatter(
    gamer,
    x='horas_jogo',
    y='cansaco',
    title='Horas Jogando x Nível de Cansaço',
    labels={
        'horas_jogo': 'Horas Jogando',
        'cansaco': 'Nível de Cansaço'
    }
)

grafico.add_scatter(
    x=gamer['horas_jogo'],
    y=gamer['previsao'],
    mode='lines',
    name='Regressão Linear'
)

grafico.add_scatter(
    x=[horas],
    y=[previsao[0]],
    mode='markers',
    name='Previsão'
)

st.plotly_chart(grafico)

st.subheader("🧠 Informações do Modelo")

coeficiente = modelo.coef_[0]
intercepto = modelo.intercept_

st.write(f"Coeficiente Angular: {coeficiente:.2f}")
st.write(f"Intercepto: {intercepto:.2f}")

st.write("### Fórmula da Regressão Linear")

st.latex("y = ax + b")

st.write(
    f"Equação encontrada: "
    f"cansaco = ({coeficiente:.2f} × horas_jogo) + ({intercepto:.2f})"
)


st.write(
    f"Equação encontrada: "
    f"cansaco = ({coeficiente:.2f} × horas_jogo) + ({intercepto:.2f})"
)

st.subheader("✅ Conclusão")

st.write("""
O modelo identificou que o aumento das horas jogando
influencia diretamente o nível de cansaço do jogador.
""")