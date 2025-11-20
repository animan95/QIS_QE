# QIS_QE
Software connecting Quantum Espresso to Qiskit, enabling Quantum Computing calculations on large materials
#  QEpy · Wannier90 · Qiskit  
### Quantum-Ready Wannier Hamiltonians from First Principles

This project provides an end-to-end workflow that takes **DFT calculations from QEpy/Quantum ESPRESSO**, constructs **maximally localized Wannier functions (MLWFs)** with **Wannier90**, and maps the resulting **tight-binding Hamiltonian** into **qubit form** using **Qiskit Nature**, enabling quantum-computing simulations of materials and molecules.

The goal is to bridge *ab-initio electronic structure* with *quantum algorithms*, supporting ground-state, excited-state, and real-time dynamics simulations on both **quantum simulators** and **real quantum hardware**.

---

##  Features

-  **QEpy-based DFT** SCF/NSCF calculations  
-  **Wannier90** generation of MLWFs and tight-binding Hamiltonians  
-  Automatic parsing of `seedname_hr.dat` → tight-binding matrix  
-  **Second-quantized Hamiltonian builder** for Qiskit Nature  
-  Qubit mapping via Jordan–Wigner or Bravyi–Kitaev  
-  **VQE**, **EOM-VQE**, **Subspace VQE**, or **Trotter time evolution**  
-  Optional execution on **real quantum hardware** (IBM, IonQ, Quantinuum)  

This framework allows quantum simulation of:
- 2D materials (graphene, MoS₂, hBN)  
- Defects and impurities  
- Molecular clusters embedded in solids  
- Simple Hubbard-like models derived directly from DFT  

---

##  Workflow Overview

```mermaid
flowchart LR
    A[DFT with QEpy] --> B[Wannier90: MLWFs]
    B --> C[Parse seedname_hr.dat]
    C --> D[Build Second-Quantized Hamiltonian]
    D --> E[Qiskit: Qubit Mapping]
    E --> F[VQE / EOM-VQE / Time Evolution]
    F --> G[Quantum Simulations or Spectroscopy]
