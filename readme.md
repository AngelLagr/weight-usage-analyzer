# Weight Usage Analyzer
Pruning and machine learning model compression techniques have been extensively studied. My goal was to investigate the importance of weights themselves, rather than just neurons or channels, to gain a finer understanding of model efficiency and complexity.

This curiosity led to this project, an open-source Python library for detailed analysis of weight usage in neural networks (compatible with PyTorch & TensorFlow).

**WeightUsageAnalyzer** is a Python toolkit for analyzing and visualizing the efficiency of neural network models in PyTorch and TensorFlow. It provides tools to quantify the actual usage of weights, estimate computational costs (FLOPs), and identify optimization opportunities, promoting an approach of "computational sobriety."

![A diagram generated by the `show()` function, depicting a network with lines of varying thicknesses and colors, illustrating well-defined neural pathways.](doc/graphs/UseCaseTorch2.png)

---

## Why Use WeightUsageAnalyzer?

As ML models become more complex, their cost in terms of computation, energy, and inference time increases. However, not all weights in a neural network contribute equally to the model's performance. I wanted to find solution and train myself in answering these questions:

* **Is my model oversized?** Identify layers or neurons that contribute little to the final decision.
* **What is the real cost of my model?** Get a concrete estimate of the computational operations (FLOPs) required for training and inference.
* **How can I make my model more efficient?** Use the generated reports to guide optimization decisions (like reducing layer sizes) without sacrificing accuracy.
* **How can I interpret the internal structure of my network?** Visualize the most important neural "pathways" to better understand your model's behavior.

The goal is to helps build models that are keep their powerness but with more efficiency and lightweight.

---

## Core Features

* **Weight Importance Analysis**: Calculates an importance metric for each weight, based on its magnitude and neuron activation.
* **Layer-wise Efficiency Reports**: Generates key statistics like weight distribution entropy, the number of "effective weights," and the percentage of low-contribution weights.
* **FLOPs Estimation**: Calculates the number of floating-point operations for training and inference, offering a standardized measure of computational cost.
* **Architecture Visualization**: Displays a diagram of the network where the thickness and color of connections represent weight importance.

---

## Quick Start

Here is a minimal example with a PyTorch model.

```python
import torch
import torch.nn as nn
import numpy as np
import core.weightusageanalyzer as wua

# 1. Create and train a model (example)
X_np = np.random.rand(100, 10).astype(np.float32)
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
# ... (model training code) ...

# 2. Analyze the model
print("--- FLOPs Analysis ---")
wua.print_flops_report(model, nb_epochs=50, dataset=X_np)

print("\n--- Weight Analysis (first layer) ---")
# Compute importance for all layers
importance_list = wua.compute_weight_importance(model, X_np)

# Generate and print the report for the first layer
importance, weights, name = importance_list[0]
report, _ = wua.generate_report(importance, weights)
print(f"Report for layer '{name}':")
wua.print_report(report)

# 3. Visualize the network
print("\n--- Network Visualization ---")
wua.show(model, X_np)
```

## Installation
Clone the repository and install locally:

```bash
git clone https://github.com/AngelLagr/WeightUsageAnalyzer.git
cd WeightUsageAnalyzer
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests to help improve the library.

---

## What's Next ?
- CNN models analysis
- more stability
---

## License

Apache 2.0 License

---

## Acknowledgements

Inspired by the need for more sustainable and interpretable deep learning.
