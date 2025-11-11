import streamlit as st
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

st.set_page_config(page_title="Simulación Estafa Piramidal", layout="wide")
st.title("Simulación de Estafa Piramidal — Resultados guardados")

st.markdown("""
**Autores:**  
- Francis Aguilar - 22243  
- Diego García - 22404  
- Ángela García - 22869  
""")

#  CARGA DE ARCHIVOS 
escenarios = {
    "Escenario base": "escenario_base.pkl",
    "Escenario alta comisión": "escenario_alta_comision.pkl",
    "Escenario optimista": "escenario_optimista.pkl"
}

opcion = st.sidebar.selectbox("Selecciona el escenario", list(escenarios.keys()))
file_path = escenarios[opcion]

try:
    with open(file_path, "rb") as f:
        params, history, G, state = pickle.load(f)
    st.success(f"Escenario '{opcion}' cargado correctamente")
except FileNotFoundError:
    st.error(f"No se encontró el archivo {file_path}.")
    st.stop()

# GRÁFICOS 
t = history["t"]

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(t, history["n_participantes"], label="Participantes totales", color="tab:blue")
    ax.plot(t, history["n_promotores"], label="Promotores", color="tab:orange")
    ax.set_title("Crecimiento de la estafa")
    ax.set_xlabel("Paso de tiempo")
    ax.set_ylabel("Número de agentes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if any(history["colapsado"]):
        t_colapso = next(tt for tt, c in zip(t, history["colapsado"]) if c)
        ax.axvline(t_colapso, linestyle="--", color="gray", alpha=0.8)
        ax.text(t_colapso, ax.get_ylim()[1]*0.9, "Colapso", rotation=90, va="top", ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(t, history["n_nuevos"], color="tab:blue", alpha=0.8)
    ax.set_title("Flujo de nuevos participantes")
    ax.set_xlabel("Paso de tiempo")
    ax.set_ylabel("Nuevos por paso")
    ax.grid(True, alpha=0.3)
    if any(history["colapsado"]):
        ax.axvline(t_colapso, linestyle="--", color="gray", alpha=0.8)
        ax.text(t_colapso, max(history["n_nuevos"] + [1]) * 0.9, "Colapso", rotation=90, va="center", ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# Distribución de ganancias
ganancias = [a["ganancia"] for _, a in G.nodes(data=True) if a["estado"] != "ignorante"]
fig, ax = plt.subplots()
ax.hist(ganancias, bins=30, color="teal", alpha=0.7)
ax.set_title("Distribución de ganancias al final de la simulación")
ax.set_xlabel("Ganancia neta de cada agente")
ax.set_ylabel("Número de agentes")
plt.tight_layout()
st.pyplot(fig)

# Ganancia total
fig, ax = plt.subplots()
ax.plot(history["t"], history["ganancia_total"], color="tab:green")
ax.set_title("Ganancia total del sistema")
ax.set_xlabel("Paso de tiempo")
ax.set_ylabel("Ganancia neta total")
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ----------------- MÉTRICAS -----------------
promedio = np.mean(ganancias)
mediana = np.median(ganancias)
max_ganancia = np.max(ganancias)
min_ganancia = np.min(ganancias)
ganadores = sum(1 for g in ganancias if g > 0)
perdedores = sum(1 for g in ganancias if g <= 0)
total = len(ganancias)

st.markdown("### Resultados finales")
st.metric("Promedio de ganancia", f"{promedio:.2f}")
st.metric("Mediana de ganancia", f"{mediana:.2f}")
st.metric("Máxima ganancia", f"{max_ganancia:.2f}")
st.metric("Mínima ganancia", f"{min_ganancia:.2f}")
st.write(f"Ganadores: {ganadores} ({ganadores/total*100:.1f}%) | Perdedores: {perdedores} ({perdedores/total*100:.1f}%)")

# Visualización de red
color_map = []
for n, a in G.nodes(data=True):
    if a["estado"] == "ignorante":
        color_map.append("lightgray")
    elif a["estado"] == "participante":
        color_map.append("orange")
    elif a["estado"] == "promotor":
        color_map.append("red")

fig, ax = plt.subplots(figsize=(7, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=30, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.3, ax=ax)
ax.set_title("Red al final de la simulación\n(gris=ignorante, naranja=participante, rojo=promotor)")
ax.axis("off")
st.pyplot(fig)
