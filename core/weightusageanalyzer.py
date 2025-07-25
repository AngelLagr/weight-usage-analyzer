# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import entropy as scipy_entropy
from ptflops import get_model_complexity_info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def compute_weight_importance_torch(model, dataset, skip_last=True):
    """
    Calculate the importance of weights for each linear layer in a PyTorch model.
    Importance = |weights| * average activation (per neuron).
    
    Parameters:
        model (torch.nn.Module): The PyTorch model.
        dataset (Union[np.ndarray, torch.Tensor]): Input data.
        skip_last (bool): Whether to skip the last layer in the computation.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray, str]]: A list of tuples containing the importance, weights, and layer names.
    """
    model.eval()
    importance_list = []
    if isinstance(dataset, torch.Tensor):
        current_input = dataset.detach().clone().float()
    else:
        current_input = torch.tensor(dataset, dtype=torch.float32)
    
    # Get list of linear layers
    linear_layers = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    skip_last = False if not skip_last else len(linear_layers) >= 2
    last_layer_name = linear_layers[-1] if skip_last else None

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            if skip_last and name == last_layer_name:
                # Skip the last layer if specified
                continue
            with torch.no_grad():
                z = layer(current_input)
                activation = torch.relu(z) if hasattr(layer, 'activation') else z
                importance = torch.abs(layer.weight) * torch.mean(activation, dim=0).unsqueeze(1)
                importance_list.append((abs(importance.numpy()).T, layer.weight.detach().numpy(), name))
                current_input = activation

    return importance_list


def compute_weight_importance_tf(model, dataset, skip_last=True):
    """
    Calculate the importance of weights for each dense layer in a TensorFlow/Keras model,
    ignoring the last layer if the model contains at least two layers.
    Importance = |weights| * average activation (per neuron).
    
    Parameters:
        model (tf.keras.Model): The TensorFlow/Keras model.
        dataset (np.ndarray): Input data.
        skip_last (bool): Whether to skip the last layer in the computation.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray, str]]: A list of tuples containing the importance, weights, and layer names.
    """
    importance_list = []
    current_input = dataset.copy()

    # Get list of dense layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    skip_last = False if not skip_last else len(dense_layers) >= 2
    layers_to_analyze = dense_layers[:-1] if skip_last else dense_layers

    for layer in layers_to_analyze:
        if not skip_last and layer == dense_layers[-1]:
            weights, biases = layer.get_weights()
            importance = np.abs(weights)       
            importance_list.append((abs(importance), weights.T, layer.name))
        else:
            weights, biases = layer.get_weights()
            z = np.dot(current_input, weights) + biases
            activations = layer.activation(z)
            importance = np.abs(weights) * np.mean(activations, axis=0)        
            importance_list.append((abs(importance), weights.T, layer.name))
            current_input = activations.numpy() if isinstance(activations, tf.Tensor) else activations

    return importance_list

def compute_weight_importance(model, dataset, skip_last=True):
    """
    Compute weight importance for a given model (TensorFlow/Keras or PyTorch).
    
    Parameters:
        model (Union[tf.keras.Model, torch.nn.Module]): The model to analyze.
        dataset (np.ndarray): Input data.
        skip_last (bool): Whether to skip the last layer in the computation.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray, str]]: A list of tuples containing the importance, weights, and layer names.
    """
    try:
        if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
            return compute_weight_importance_tf(model, dataset,skip_last=skip_last)
    except ImportError:
        pass

    try:
        if isinstance(model, torch.nn.Module):
            return compute_weight_importance_torch(model, dataset,skip_last=skip_last)
    except ImportError:
        pass

    raise TypeError("Unrecognized model type: must be a TensorFlow/Keras or PyTorch model.")


def compute_entropy(normalized_importance):
    """
    Compute the entropy of the normalized importance values.
    
    Parameters:
        normalized_importance (np.ndarray): Normalized importance values.
        
    Returns:
        float: The computed entropy.
    """
    return scipy_entropy(normalized_importance + 1e-8)

def compute_topk_coverage(importance_flat, k=0.9):
    """
    Calculate the coverage of the top k% of weights by importance.
    
    Parameters:
        importance_flat (np.ndarray): Flattened importance values.
        k (float): Proportion of weights to consider (between 0 and 1).
        
    Returns:
        float: The coverage of the top k% of weights.
    """
    sorted_importance = np.sort(importance_flat)[::-1]
    cumsum = np.cumsum(sorted_importance)
    threshold = k * np.sum(sorted_importance)
    topk_count = np.searchsorted(cumsum, threshold) + 1
    return topk_count / len(importance_flat)

def plot_importance_histogram(normalized_importance, entropy_val=None):
    """
    Plot a histogram of the normalized importance values.
    
    Parameters:
        normalized_importance (np.ndarray): Normalized importance values.
        entropy_val (Optional[float]): Entropy value for the title.
    """

    plt.figure(figsize=(10, 4))

    # Remove NaN values from normalized importance, may lead to issues in histogram plotting
    normalized_importance = np.asarray(normalized_importance)
    normalized_importance = normalized_importance[~np.isnan(normalized_importance)]

    plt.hist(normalized_importance, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Weight Contribution Distribution\nEntropy = {entropy_val:.4f}" if (entropy_val is not None) else "Weight Contribution Distribution")
    plt.xlabel("Normalized Importance")
    plt.ylabel("Number of Weights")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_report(importance, weights):
    """
    Generate a report based on the computed importance and weights.
    
    Parameters:
        importance (np.ndarray): Importance values.
        weights (np.ndarray): Weights of the model.
        
    Returns:
        Tuple[Dict[str, float], np.ndarray]: A report dictionary and normalized importance values.
    """
    importance_flat = importance.flatten()
    normalized_importance = importance_flat / np.sum(importance_flat)
    ent = compute_entropy(normalized_importance)
    if np.isnan(ent) or np.isinf(ent):
        ent = 0.0
    topk_90 = compute_topk_coverage(importance_flat, k=0.9)
    low_contrib = np.sum(importance_flat < 1e-2) / len(importance_flat)

    report = {
        "total_weights": np.prod(weights.shape),
        "nodes": weights.shape[0],
        "entropy": float(ent),
        "effective_weights": float(np.exp(ent)),
        "top_90_percent_contrib": float(topk_90),
        "low_contrib_percentage": float(low_contrib * 100),
    }
    return report, normalized_importance

def print_report(report):
    """
    Print the weight usage report.
    
    Parameters:
        report (Dict[str, float]): The report dictionary.
    """
    print("\n📊 Weight Usage Report:")
    print(f"Total number of weights: {report['total_weights']}")
    print(f"Number of nodes (neurons): {report['nodes']}")
    print(f"Entropy (measure of uncertainty): {report['entropy']:.4f}")
    print(f"Effective weights (active weights count): {report['effective_weights']:.4f}")
    print(f"Contribution of top 90% of weights: {report['top_90_percent_contrib'] * 100:.2f}%")
    print(f"Percentage of low weights (<1e-2): {report['low_contrib_percentage']:.2f}%")


def estimate_flops(model, nb_epochs, dataset):
    """
    Estimate FLOPs for training (forward + backward) and inference (forward) for a TensorFlow/Keras or PyTorch model.
    
    Parameters:
        model (Union[tf.keras.Model, torch.nn.Module]): The model to analyze.
        nb_epochs (int): Number of training epochs.
        dataset (np.ndarray): The dataset used for estimation.
        
    Returns:
        Tuple[int, int]: Estimated FLOPs for training and inference.
    """
    def get_flops_tf_keras(model: tf.keras.Model) -> int:
        from tensorflow.python.framework import convert_to_constants
        concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype))
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'none'

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops

    nb_samples = len(dataset)

    # If the model is a TensorFlow/Keras model, we use TensorFlow's profiling tools to estimate FLOPs
    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        flops_per_sample_forward = get_flops_tf_keras(model)
        flops_training = 2 * flops_per_sample_forward * nb_epochs * nb_samples
        flops_inference = flops_per_sample_forward * nb_samples
        return flops_training, flops_inference

    # If the model is a PyTorch model, we use ptflops to estimate FLOPs
    elif isinstance(model, torch.nn.Module):
        input_shape = (dataset.shape[1],)
        macs, _ = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False, verbose=False)
        flops_per_sample_forward = 2 * macs
        flops_training = 2 * flops_per_sample_forward * nb_epochs * nb_samples
        flops_inference = flops_per_sample_forward * nb_samples
        return flops_training, flops_inference

    else:
        raise TypeError("Unsupported model type for FLOPs estimation (neither TensorFlow/Keras nor PyTorch).")



def print_flops_report(model, nb_epochs, dataset):
    """
    Print the FLOPs estimation report for training and inference.
    
    Parameters:
        model (Union[tf.keras.Model, torch.nn.Module]): The model to analyze.
        nb_epochs (int): Number of training epochs.
        dataset (np.ndarray): The dataset used for estimation.
    """
    train_flops, infer_flops = estimate_flops(model, nb_epochs, dataset)
    nb_samples = len(dataset)
    print("\n🧮 FLOPs Estimation:")
    print(f" - Training ({nb_epochs} epochs, {nb_samples} samples): {train_flops:,} operations")
    print(f" - Inference ({nb_samples} samples): {infer_flops:,} operations")


def show(model, dataset):
    """
    Visualize the neural network structure with weight importance.
    
    Parameters:
        model (Union[tf.keras.Model, torch.nn.Module]): The model to visualize.
        dataset (np.ndarray): Input data.
    """

    importance_list = compute_weight_importance(model, dataset, skip_last=False)    

    layer_positions = []
    layer_names = []

    # Input layer
    input_dim = dataset.shape[1]
    input_layer = [f"Input {i}" for i in range(input_dim)]
    layer_positions.append(input_layer)
    layer_names.append("Input")

    # Hidden layers
    for (importance, weights, name) in importance_list:
        importance = np.mean(importance, axis=0)
        layer = [f"{name}\nN{i}" for i in range(len(importance))]
        layer_positions.append(layer)
        layer_names.append(name)

    # Figure
    _, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    spacing_x = 3
    spacing_y = 1.5

    positions = {}  # id → (x, y)
    for i, layer in enumerate(layer_positions):
        x = i * spacing_x
        total = len(layer)
        for j, label in enumerate(layer):
            y = -j * spacing_y + (total - 1) * spacing_y / 2
            ax.add_patch(patches.Circle((x, y), 0.2, color="lightgrey", zorder=2))
            ax.text(x, y, label.split('\n')[0], ha='center', va='center', fontsize=8)
            positions[(i, j)] = (x, y)

    # Weights visualization
    for i in range(len(layer_positions) - 1):
        src_size = len(layer_positions[i])
        dst_size = len(layer_positions[i + 1])
        weights = importance_list[i][1].T 
        for src in range(min(src_size, weights.shape[0])):
            for dst in range(min(dst_size, weights.shape[1])):
                weight = weights[src][dst]
                color = 'red' if abs(weight) > 1e-1 else 'lightgray'
                linewidth = min(max(abs(weight)*3, 0.2), 3)
                x1, y1 = positions[(i, src)]
                x2, y2 = positions[(i + 1, dst)]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=1)

    plt.title("Visualization of your Neural Network with Weight Importance")
    plt.show()
