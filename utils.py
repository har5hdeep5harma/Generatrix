import torch
from collections import OrderedDict
import matplotlib.pyplot as plt

def vector_to_target_weights(generated_vector, param_dict):

    target_weights = OrderedDict()
    current_pos = 0
    for name, shape in param_dict.items():
        num_params = torch.prod(torch.tensor(shape)).item()
        param_slice = generated_vector[current_pos : current_pos + num_params]
        target_weights[name] = param_slice.view(shape)
        current_pos += num_params
    
    if current_pos != len(generated_vector):
        raise ValueError("Mismatch between generated vector length and total parameters in target network.")
        
    return target_weights

def plot_training_results(train_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('result_curves.png')
    print("\nTraining curves saved to 'result_curves.png'")
