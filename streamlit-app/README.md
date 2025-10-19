## GNN Failure Prediction (JAX + Streamlit)

DEPLOYED ON [Streamlit Cloud](https://gnn-challenge.streamlit.app)

This app runs a **Graph Convolutional Network (GCN)** implemented in **JAX** inside a **Streamlit** interface.
It trains a simple two-layer GCN on either the **Karate Club** dataset or a **Stochastic Block Model (SBM)** and visualizes node-level predictions.

These datasets were selected for their simplicity and interpretability, making them ideal for demonstrating GNN concepts.

* **Karate Club:** A real-world social network of 34 members split into two communities. The task is to predict each member’s club affiliation based on their connections.
* **Stochastic Block Model (SBM):** A synthetic graph generated with multiple communities where edges are denser within groups than between them, simulating clustered network structures.

GNNs are well-suited for such **node classification** problems, as they learn representations from both **node features** and **graph connectivity**.


### How to Run

```bash
pip install streamlit jax jaxlib networkx matplotlib numpy

streamlit run app.py
```

Then open the local URL shown in the terminal.


### App Features

* Choose between **Karate Club** or **Synthetic SBM** dataset
* Adjust hidden dimension, learning rate, epochs, dropout, and SBM parameters
* Click **Run Model** to train and view results
* Interactive graph visualization and node probe for predictions


### Model Summary

* **Architecture:** Two-layer GCN with ReLU and Dropout
* **Framework:** JAX (GPU-ready)
* **Loss:** Cross-entropy on labeled nodes
* **Output:** Node-wise class prediction and failure probability


### Notes

* Identity matrix used as input features (no external node attributes)
* Early stopping if validation accuracy stalls for 50 epochs
* Accuracy may vary since a **basic setup** was intentionally used for clarity and reproducibility

---
**Author:** Karan Anand  
**Date:** 19 October 2025  