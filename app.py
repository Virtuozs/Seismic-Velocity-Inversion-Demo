import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from optimizers import gradient_descent, quasi_newton

st.set_page_config(
    page_title="Seismic Inversion with Gradient Descent",
    layout="wide"
)

st.title("Seismic Inversion using Gradient-Based Optimization")

st.markdown("""
This application studies **convergence behavior of Gradient Descent and Quasi-Newton methods**
in a **physics-inspired seismic inversion problem**.
""")

# Sidebar
st.sidebar.header("Model Configuration")

N = st.sidebar.slider("Number of Layers", 2, 20, 5)
layer_thickness = st.sidebar.number_input("Layer Thickness (m)", 1.0, 100.0, 10.0)

st.sidebar.subheader("True Velocity Model (m/s)")
v_true = np.array([
    st.sidebar.number_input(
        f"v_true[{i+1}]",
        500, 5000,
        1500 + i * 300,
        step=50
    )
    for i in range(N)
])

noise = st.sidebar.slider("Noise Level σ", 0.0, 0.05, 0.0, 0.005)

st.sidebar.header("Optimization Settings")

method = st.sidebar.selectbox(
    "Optimization Method",
    ["Gradient Descent", "Quasi-Newton"]
)

learning_rate = st.sidebar.number_input("Learning Rate γ", 0.0001, 1000.0, 10.0)
iterations = st.sidebar.slider("Iterations", 10, 500, 100, 10)
initial_velocity = st.sidebar.number_input("Initial Velocity (m/s)", 500, 6000, 3000)
lambda_reg = st.sidebar.slider("Regularization λ", 0.0, 1.0, 0.0, 0.01)

# Data Generation based on Input
d = np.ones(N) * layer_thickness
T_true = np.sum(d / v_true)
T_obs = T_true * (1 + np.random.normal(0, noise))

v0 = np.ones(N) * initial_velocity


# Optimizations
if method == "Gradient Descent":
    v_est, history = gradient_descent(
        v0, d, T_obs, learning_rate, iterations, lambda_reg
    )
else:
    v_est, history = quasi_newton(
        v0, d, T_obs, learning_rate, iterations, lambda_reg
    )

# Visualizations
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(v)")
    ax.set_title("Cost Convergence")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(v_true, label="True Velocity", marker="o")
    ax2.plot(v_est, label="Estimated Velocity", marker="x")
    ax2.legend()
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.grid(True)
    st.pyplot(fig2)


st.subheader("Diagnostics")

st.write(f"Observed Travel Time: `{T_obs:.6f}`")
st.write(f"Predicted Travel Time: `{np.sum(d / v_est):.6f}`")
st.write(f"Final Cost: `{history[-1]:.6e}`")

if history[-1] > history[0]:
    st.error("Divergence detected.")
elif history[-1] < 1e-6:
    st.success("Convergence achieved.")
else:
    st.warning("Partial convergence.")
