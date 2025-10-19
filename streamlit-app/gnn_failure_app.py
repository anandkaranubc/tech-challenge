import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------
st.set_page_config(page_title="GNN Failure Prediction", layout="wide")
st.title("GNN Failure Prediction (JAX)")

st.write(
    "This demo trains a simple two-layer Graph Convolutional Network (GCN) "
    "implemented in JAX on either the Karate Club dataset or a synthetic "
    "Stochastic Block Model (SBM)."
)

# Display JAX device info
device = jax.devices()[0]
st.info(f"Using JAX device: {device.platform.upper()}")

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def normalized_adjacency(G, self_loops=True):
    """Compute the symmetric normalized adjacency matrix."""
    A = nx.to_numpy_array(G, dtype=np.float32)
    if self_loops:
        A += np.eye(A.shape[0])
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1) + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return jnp.array(A_norm)

@st.cache_data
def load_karate():
    """Karate Club graph — binary classification task."""
    G = nx.karate_club_graph()
    X = np.eye(G.number_of_nodes(), dtype=np.float32)
    y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes()])
    A = normalized_adjacency(G)
    return jnp.array(X), A, jnp.array(y), G

@st.cache_data
def load_sbm(nc=4, npn=250, p_in=0.08, p_out=0.005):
    """Generate a stochastic block model for community detection."""
    sizes = [npn] * nc
    P = np.full((nc, nc), p_out, dtype=np.float32)
    np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model(sizes, P, seed=42)
    X = np.eye(G.number_of_nodes(), dtype=np.float32)
    y = np.concatenate([[i] * npn for i in range(nc)])
    A = normalized_adjacency(G)
    return jnp.array(X), A, jnp.array(y), G

# ------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------
with st.sidebar:
    st.header("Model Settings")
    dataset = st.selectbox("Dataset", ["Karate Club", "Synthetic SBM"])
    seed = st.number_input("Seed", 42)
    hidden = st.number_input("Hidden dimension", 16)
    lr = st.number_input("Learning rate", 0.01)
    epochs = st.number_input("Epochs", 800)
    dropout_rate = st.slider("Dropout", 0.0, 0.7, 0.2)

    if dataset == "Synthetic SBM":
        nc = st.number_input("Communities", 4)
        npn = st.number_input("Nodes per community", 250)
        p_in = st.slider("p_in", 0.0, 0.2, 0.08)
        p_out = st.slider("p_out", 0.0, 0.05, 0.005)
    else:
        nc = npn = p_in = p_out = None

    run_button = st.button("Run Model")

# ------------------------------------------------------
# Data loading
# ------------------------------------------------------
if dataset == "Karate Club":
    X, A, y_true, G = load_karate()
else:
    X, A, y_true, G = load_sbm(nc, npn, p_in, p_out)

n, f = X.shape
c = int(y_true.max()) + 1
idx = np.arange(n)
np.random.default_rng(seed).shuffle(idx)

# Train / validation / test split (60/20/20)
n_tr, n_val = int(0.6 * n), int(0.2 * n)
train, val, test = (
    jnp.array(idx[:n_tr]),
    jnp.array(idx[n_tr:n_tr + n_val]),
    jnp.array(idx[n_tr + n_val:])
)

Y = jax.nn.one_hot(y_true, c)

# ------------------------------------------------------
# Model definition
# ------------------------------------------------------
def glorot(key, shape):
    """Glorot/Xavier uniform initialization."""
    limit = jnp.sqrt(6.0 / sum(shape))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def gcn(A, X, W):
    """Single graph convolution layer."""
    return A @ (X @ W)

def forward(params, A, X, rng_key=None, dropout=0.2, train=True):
    """Forward pass through two-layer GCN."""
    W1, W2 = params
    H = jnp.maximum(0, gcn(A, X, W1))  # ReLU activation
    if train and rng_key is not None and dropout > 0:
        mask = jax.random.bernoulli(rng_key, 1 - dropout, H.shape)
        H = H * mask / (1 - dropout)
    return gcn(A, H, W2)

def loss(params, A, X, Y, idx):
    """Cross-entropy loss on labeled nodes."""
    logits = forward(params, A, X, train=True)[idx]
    Y_ = Y[idx]
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(Y_ * logp, axis=1))

@jax.jit
def accuracy(params, A, X, y, idx):
    logits = forward(params, A, X, train=False)
    return (jnp.argmax(logits, 1)[idx] == y[idx]).mean()

@jax.jit
def step(params, A, X, Y, idx, lr, rng):
    grads = jax.grad(loss)(params, A, X, Y, idx)
    new_params = [w - lr * dw for w, dw in zip(params, grads)]
    return new_params

@jax.jit
def predict(params, A, X):
    logits = forward(params, A, X, train=False)
    probs = jax.nn.softmax(logits)
    return probs, jnp.argmax(probs, 1)

# ------------------------------------------------------
# Model training
# ------------------------------------------------------
if run_button:
    st.subheader("Training Model")
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params = [glorot(k1, (f, hidden)), glorot(k2, (hidden, c))]

    with st.spinner("Training in progress..."):
        best_params = [w.copy() for w in params]
        best_acc, patience = 0.0, 0

        for e in range(int(epochs)):
            key, subkey = jax.random.split(key)
            params = step(params, A, X, Y, train, lr, subkey)

            # Early stopping on validation accuracy
            if e % 10 == 0:
                val_acc = float(accuracy(params, A, X, y_true, val))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = [w.copy() for w in params]
                    patience = 0
                else:
                    patience += 1
                if patience > 50:
                    break

        params = best_params
        test_acc = float(accuracy(params, A, X, y_true, test))
        st.success(f"Test accuracy: {test_acc:.3f}")

        probs, logits = predict(params, A, X)
        st.session_state["params"] = params
        st.session_state["probs"] = probs
        st.session_state["logits"] = logits
        st.session_state["fail_prob"] = probs[:, 1] if c == 2 else probs[:, 0]
        st.session_state["A"] = A
        st.session_state["G"] = G
        st.session_state["y_true"] = y_true

# ------------------------------------------------------
# Visualization and probing
# ------------------------------------------------------
if "params" in st.session_state:
    st.subheader("Graph View")

    logits = st.session_state["logits"]
    fail_prob = st.session_state["fail_prob"]
    G = st.session_state["G"]
    y_true = st.session_state["y_true"]

    # Plot only first 200 nodes for large graphs
    subG = G if len(G) <= 200 else G.subgraph(list(range(200)))
    pos = nx.spring_layout(subG, seed=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:green" if int(logits[i]) == 0 else "tab:red" for i in subG.nodes()]
    sizes = [100 + 300 * float(fail_prob[i]) for i in subG.nodes()]
    nx.draw(subG, pos, node_color=colors, node_size=sizes, with_labels=len(G) <= 60, ax=ax)
    st.pyplot(fig)

    st.subheader("Probe Node")
    i = st.number_input("Node ID", 0, len(G) - 1, 0, key="probe_node")
    st.write(
        f"Node {i}: failure prob={fail_prob[i] * 100:.2f}%, "
        f"pred={int(logits[i])}, true={int(y_true[i])}"
    )
else:
    st.info("Train the model first by clicking 'Run Model' in the sidebar.")

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.caption("Two-layer GCN (ReLU + Dropout) implemented in JAX | Datasets: Karate Club / SBM | Author: Karan Anand")
