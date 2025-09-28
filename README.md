# Generatrix  

> *"The amateur trains until he gets it right. The professional trains until he can't get it wrong."*

From-scratch **PyTorch implementation** of a **Hypernetwork**, inspired by the [ICLR 2017 paper “Hypernetworks” by Ha, Dai, and Le](https://arxiv.org/abs/1609.09106).

---

## The Paradigm Shift  

Conventional deep learning is **parameter-inefficient**.  
A model with millions of parameters literally stores millions of floating-point numbers.

**Generatrix** explores a more profound approach:

> **Can a smaller “Hyper” network learn to generate the parameters for a larger “Target” network on demand?**

This repository demonstrates this is not only possible, but practical.  

- We train a small **Multi-Layer Perceptron (MLP)** as a **Hypernetwork**.  
- Its sole task: output the entire **weight and bias tensors** for a **Convolutional Neural Network (CNN)**.  
- The CNN is then immediately used for **image classification on CIFAR-10**.  

Gradients from the CNN’s classification loss are **backpropagated through the generated weights** and into the Hypernetwork itself.  
The Hypernetwork never “sees” the image data; it only learns how to **construct a machine (the CNN)** that can.

---

## Why This Matters  

- **Meta-Learning**  
  Optimizes the **weight-generating process**, not just the weights themselves.  

- **Model Compression**  
  Only the small Hypernetwork is stored, not the millions of parameters in the target CNN.  

- **Dynamic Architectures**  
  By conditioning the Hypernetwork on different inputs, it could generate specialized CNNs for different tasks **on the fly** without retraining.

This project is a **functional proof-of-concept** for a more abstract and powerful way of building intelligent systems.

---

## How to Run  

1. Clone the repository:

    ```bash
    git clone https://github.com/har5hdeep5harma/Generatrix.git
    cd Generatrix
    ```
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:

    ```bash
    python main.py
    ```
For best results, run on a system with a CUDA-enabled GPU or use a cloud environment.

## Results & Proof of Concept

The final training curves provide definitive proof that the Hypernetwork learned to generate a functional and effective classifier:

- Training loss steadily decreases
- Accuracy on unseen test data consistently improves and stabilizes
- Demonstrating true generalization and the success of the core concept

<img src="https://raw.githubusercontent.com/har5hdeep5harma/Generatrix/refs/heads/main/result_curves.png" alt="Training Curves" width="600"/>

## Reference

Ha, D., Dai, A. M., & Le, Q. V. (2017). [HyperNetworks](https://arxiv.org/abs/1609.09106)
. International Conference on Learning Representations (ICLR).

## License

This project is licensed under the [MIT License](LICENSE).