# 🌌 N-Body Simulation with AI Trajectory Estimation

A **physics-accurate N-body gravitational simulator** with an **AI-assisted trajectory prediction layer**, designed to study **chaotic orbital dynamics**, **binary formation**, **escapes**, and **long-term evolution** of multi-body systems.

This project combines **classical Newtonian mechanics** with **machine learning** to estimate how trajectories evolve over time given initial conditions — especially in regimes where numerical chaos dominates.

---

## 🚀 Project Motivation

Classical N-body systems are:

* **Highly chaotic** for N ≥ 3
* Extremely **sensitive to initial conditions**
* Computationally expensive for long-term prediction

This repository explores a hybrid approach:

> **Use physics for correctness, AI for foresight.**

The simulator computes accurate short-term dynamics, while an AI model learns to:

* Forecast long-term trajectory drift
* Predict qualitative outcomes (binary formation, escape, clustering)
* Approximate system evolution faster than brute-force integration

---

## 🧠 Core Features

### 🔭 Physics Engine (developed from Scratch)

* Full **Newtonian N-body gravity**
* Mass-weighted **center-of-mass (COM)** correction
* Energy-consistent dynamics (kinetic + potential)
* Pairwise interaction tracking
* Supports:

  * Bound orbits
  * Binary systems
  * Flybys & escapes
  * Temporary clusters
*Results are never accurate but approximations*

### ⚖️ Energy & Stability Analysis (Complete details written later)

* Pairwise reduced two-body energy detection
* Bound / unbound classification:

  * **E < 0** → bound system
  * **E = 0** → parabolic escape
  * **E > 0** → hyperbolic escape
* Group energy evaluation for multi-body clusters
* Detection of:

  * Stable binaries
  * Hierarchical systems
  * Chaotic but bounded motion

### 🧬 AI Trajectory Estimation (In Development)

* Learns from **physics-generated trajectories**
* Predicts:

  * Future positions & velocities
  * Orbital drift under chaos
  * Likely system outcomes
* Intended roles:

  * Surrogate model for long-term evolution
  * Error-corrector for numerical integration
  * Fast estimator for parameter sweeps

*(AI layer is modular and optional — physics simulation works standalone.)*

---

## 🧪 Physics Assumptions

* Newtonian gravity (velocity Verlet) (relativistic effects not included)
* Point masses
* No collisions by default (merging optional)
* Inertial reference frame with COM correction

These choices ensure:

* Energy interpretability
* Clean binary detection
* Stable long-term integration

---

## 🛠️ Tech Stack

* **Python** – core simulation
* **NumPy** – vectorized physics
* **Pytorch** – for efficient collision Detection and AI implementation
* **Scipy** – for basic collision Detection
* **Taichi** – Gpu accelerated collision detection (Not completed as very less compatibility with numpy used pytorch instead) (last implementations available)
* **JAX** – experimental accelerated & differentiable backend for collision detection (Not completed as so only linux, used pytorch instead) (last implementations available)
* **Arcade** – For GUI

---

## 🧠 Example Phenomena Studied

* Binary planet formation
* Temporary gravitational capture
* Three-body chaos
* Escaping bodies
* Long-term orbital instability
* Energy exchange in close encounters

---

## 🎯 Long-Term Goals

* AI-assisted **chaos forecasting**
* Differentiable physics via JAX
* Dataset generation for ML research
* Educational sandbox for orbital mechanics
* Extendable to:

  * Collisions & mergers
  * Soft-body gravity
  * Relativistic corrections

---

## 📊 Intended Use Cases

* Computational astrophysics research
* Physics-informed ML experimentation
* Chaos theory exploration
* Educational demonstrations
* Fast parameter sweeps for orbital systems

---

## ⭐ A Note on Chaos

> *In an N-body system, precision guarantees correctness — but never predictability.*

This project exists to explore that boundary.

Contributions, experiments, and discussions are welcome.
If you're interested in **physics-informed AI**, **orbital mechanics**, or **chaotic systems**, feel free to fork and explore.
