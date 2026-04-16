import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
import pennylane as qml

st.set_page_config(page_title="Iris: ML vs DL vs QML", layout="wide")

CLASSES = ["setosa", "versicolor", "virginica"]
FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]
COLORS = {"ML": "#4C8EDA", "DL": "#F4845F", "QML": "#6DBF8B"}

def make_cm_fig(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig

def metrics_row(y_true, y_pred, acc):
    f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
    prec = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
    rec = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc}%")
    c2.metric("F1 Score", f"{f1}%")
    c3.metric("Precision", f"{prec}%")
    c4.metric("Recall", f"{rec}%")
    return f1, prec, rec

@st.cache_data
def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte

@st.cache_data
def run_ml():
    Xtr, Xte, ytr, yte = get_data()
    configs = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM (RBF)", SVC(kernel="rbf", probability=True, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]
    out = {}
    for name, clf in configs:
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        imp = getattr(clf, "feature_importances_", None)
        coef = np.abs(clf.coef_).mean(0) if hasattr(clf, "coef_") else None
        out[name] = {
            "acc": round(accuracy_score(yte, yp) * 100, 2),
            "cm": confusion_matrix(yte, yp),
            "y_pred": yp,
            "y_true": yte,
            "imp": imp,
            "coef": coef,
        }
    return out

@st.cache_data
def run_dl():
    Xtr, Xte, ytr, yte = get_data()
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(Xtr, ytr, epochs=100, batch_size=8, validation_split=0.15, verbose=0)
    yp = np.argmax(model.predict(Xte, verbose=0), axis=1)
    return {
        "acc": round(accuracy_score(yte, yp) * 100, 2),
        "cm": confusion_matrix(yte, yp),
        "y_pred": yp,
        "y_true": yte,
        "history": h.history,
    }

@st.cache_data
def run_qml():
    Xtr, Xte, ytr, yte = get_data()
    n_tr, n_te = 45, 20
    Xtr2, ytr2 = Xtr[:n_tr], ytr[:n_tr]
    Xte2, yte2 = Xte[:n_te], yte[:n_te]
    lo, hi = Xtr2.min(0), Xtr2.max(0)
    Xtr2 = (Xtr2 - lo) / (hi - lo + 1e-8) * np.pi
    Xte2 = (Xte2 - lo) / (hi - lo + 1e-8) * np.pi

    dev = qml.device("default.qubit", wires=4)

    def feature_map(x):
        qml.AngleEmbedding(x, wires=range(4), rotation="X")

    @qml.qnode(dev)
    def kcirc(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(4))

    Ktr = np.array([[float(kcirc(a, b)[0]) for b in Xtr2] for a in Xtr2])
    Kte = np.array([[float(kcirc(a, b)[0]) for b in Xtr2] for a in Xte2])

    svc = SVC(kernel="precomputed")
    svc.fit(Ktr, ytr2)
    yp = svc.predict(Kte)

    return {
        "acc": round(accuracy_score(yte2, yp) * 100, 2),
        "cm": confusion_matrix(yte2, yp),
        "y_pred": yp,
        "y_true": yte2,
        "Ktr": Ktr,
        "n_tr": n_tr,
        "n_te": n_te,
    }

st.title("Iris Classification: ML · DL · Quantum ML")
st.caption("The same 3-class classification task solved three different ways on the Iris dataset.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["ML Models", "Deep Learning", "Quantum ML", "Comparison"])

with tab1:
    st.subheader("Classical Machine Learning")
    st.caption("Three classical algorithms trained on 105 samples, tested on 45.")
    with st.spinner("Training ML models..."):
        ml = run_ml()

    for name, r in ml.items():
        st.markdown(f"#### {name}")
        metrics_row(r["y_true"], r["y_pred"], r["acc"])
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.pyplot(make_cm_fig(r["cm"]))
        with c2:
            if r["imp"] is not None:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                bars = ax.barh(FEATURES, r["imp"], color="#4C8EDA", height=0.5)
                ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
                ax.set_title("Feature Importance", fontsize=9)
                ax.set_xlabel("Importance score", fontsize=8)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif r["coef"] is not None:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                bars = ax.barh(FEATURES, r["coef"], color="#F4845F", height=0.5)
                ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
                ax.set_title("Avg |Coefficient| per Feature", fontsize=9)
                ax.set_xlabel("Mean absolute weight", fontsize=8)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("RBF kernel maps input to an implicit high-dimensional space — feature importances are not directly available. The kernel compares samples by similarity.")
                fig, ax = plt.subplots(figsize=(5, 2.8))
                class_acc = [
                    (r["cm"][i, i] / r["cm"][i].sum() * 100) for i in range(3)
                ]
                ax.bar(CLASSES, class_acc, color="#4C8EDA", width=0.4)
                ax.set_ylim(0, 110)
                ax.set_title("Per-Class Accuracy", fontsize=9)
                ax.set_ylabel("Accuracy (%)", fontsize=8)
                for i, v in enumerate(class_acc):
                    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        st.divider()

with tab2:
    st.subheader("Deep Learning — Multilayer Perceptron")
    st.caption("Two hidden layers (16→8→3) trained with Adam optimizer for 100 epochs.")
    with st.spinner("Training neural network..."):
        dl = run_dl()

    metrics_row(dl["y_true"], dl["y_pred"], dl["acc"])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.pyplot(make_cm_fig(dl["cm"]))
    with c2:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].plot(dl["history"]["loss"], color="#4C8EDA", label="Train")
        axes[0].plot(dl["history"]["val_loss"], color="#F4845F", linestyle="--", label="Val")
        axes[0].set_title("Loss", fontsize=9)
        axes[0].legend(fontsize=7)
        axes[0].set_xlabel("Epoch", fontsize=8)
        axes[0].tick_params(labelsize=7)
        axes[1].plot(dl["history"]["accuracy"], color="#4C8EDA", label="Train")
        axes[1].plot(dl["history"]["val_accuracy"], color="#F4845F", linestyle="--", label="Val")
        axes[1].set_title("Accuracy", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].set_xlabel("Epoch", fontsize=8)
        axes[1].tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Architecture details"):
        st.code("Dense(16, relu) → Dense(8, relu) → Dense(3, softmax)\nLoss: sparse_categorical_crossentropy | Optimizer: Adam | Epochs: 100")

with tab3:
    st.subheader("Quantum ML — Quantum Kernel SVM")
    st.caption("A quantum kernel encodes each sample into a 4-qubit circuit state. The kernel matrix measures quantum state overlap, then feeds a classical SVM.")
    st.info("⚡ Computed on a 45-sample training subset (quantum circuit simulation scales quadratically with samples).")
    with st.spinner("Computing quantum kernel matrix — this may take ~30–60 seconds on first run..."):
        qr = run_qml()

    metrics_row(qr["y_true"], qr["y_pred"], qr["acc"])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.pyplot(make_cm_fig(qr["cm"]))
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        im = ax.imshow(qr["Ktr"], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Kernel value")
        ax.set_title(f"Quantum Kernel Matrix ({qr['n_tr']}×{qr['n_tr']} training set)", fontsize=9)
        ax.set_xlabel("Sample index", fontsize=8)
        ax.set_ylabel("Sample index", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("How the circuit works"):
        st.markdown("""
        **Circuit per kernel entry k(x₁, x₂):**
        1. Apply `AngleEmbedding(x₁)` — encodes 4 features as Rx rotations on 4 qubits
        2. Apply `AngleEmbedding†(x₂)` — the adjoint (inverse) circuit
        3. Measure probability of |0000⟩ state → this is the kernel value

        High value = samples are "close" in quantum state space. The SVM uses these similarity scores instead of raw features.
        """)

with tab4:
    st.subheader("Comparison")
    with st.spinner("Loading all results..."):
        ml = run_ml()
        dl = run_dl()
        qr = run_qml()

    all_names = list(ml.keys()) + ["MLP (Deep Learning)", "Quantum Kernel SVM"]
    all_accs = [r["acc"] for r in ml.values()] + [dl["acc"], qr["acc"]]
    all_types = ["ML", "ML", "ML", "DL", "QML"]
    bar_colors = [COLORS[t] for t in all_types]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(all_names, all_accs, color=bar_colors, width=0.5, edgecolor="white")
    ax.set_ylim(50, 110)
    ax.set_ylabel("Test Accuracy (%)", fontsize=9)
    ax.set_title("Accuracy across all models", fontsize=10)
    for bar, val in zip(bars, all_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.xticks(rotation=15, ha="right")
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=COLORS[k], label=k) for k in COLORS]
    ax.legend(handles=legend_els, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    df = pd.DataFrame({
        "Model": all_names,
        "Type": all_types,
        "Accuracy (%)": all_accs,
        "Training Samples": [105, 105, 105, 105, 45],
        "Interpretable": ["Yes", "No", "Yes", "Partial", "No"],
        "Speed": ["Fast", "Fast", "Fast", "Medium", "Slow"],
        "Scales to big data": ["Well", "Well", "Well", "Best", "Not yet"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### What each approach brings")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### 🌳 Classical ML")
        st.markdown("""
**What it does:** Directly fits rules/boundaries to structured features.

**Strengths:**
- Very fast to train
- Random Forest reveals which features matter most
- Works great on small, clean tabular data like Iris

**Weakness:** Needs hand-crafted features for complex data like images or text.
        """)
    with c2:
        st.markdown(f"#### 🧠 Deep Learning")
        st.markdown("""
**What it does:** Learns hierarchical representations through layers of neurons.

**Strengths:**
- Learns complex non-linear patterns automatically
- Dominates images, text, speech at scale

**Weakness:** Overkill here — needs more data to outshine classical ML. Training curves show it converges fine but doesn't gain much over simpler models on Iris.
        """)
    with c3:
        st.markdown(f"#### ⚛️ Quantum ML")
        st.markdown("""
**What it does:** Encodes data into quantum states; uses quantum state overlap as a similarity kernel.

**Strengths:**
- Can represent exponentially large feature spaces with few qubits
- Theoretically may capture correlations classical kernels miss

**Weakness:** Currently slow (simulated on classical hardware), limited by NISQ-era noise, and no proven quantum advantage on classical datasets yet.
        """)

    st.markdown("---")
    best_idx = int(np.argmax(all_accs))
    best_name = all_names[best_idx]
    best_acc = all_accs[best_idx]

    st.success(f"""
**🏆 Best performer on Iris: {best_name} — {best_acc}%**

For this dataset, classical ML wins. Iris is small (150 samples), clean, and nearly linearly separable — exactly where classical algorithms like Logistic Regression and Random Forest thrive. 

Deep Learning matches accuracy but adds unnecessary complexity. Quantum ML shows promise as a concept, but on classical hardware simulation with small training subsets, it can't yet compete.

**The takeaway:** Match the tool to the problem. Classical ML for structured tabular data. Deep Learning for high-dimensional unstructured data at scale. Quantum ML — watch this space; the advantage will come as quantum hardware matures.
    """)
