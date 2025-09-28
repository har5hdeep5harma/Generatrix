import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.target_network import TargetNet
from models.hypernetwork import HyperNet
from utils import vector_to_target_weights, plot_training_results

def main():
    print(f"Using device: {config.DEVICE}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    target_net_template = TargetNet(config.IMG_CHANNELS, config.NUM_CLASSES)
    param_dict = target_net_template.get_param_dict()
    total_target_params = sum(torch.prod(torch.tensor(s)).item() for s in param_dict.values())
    
    hyper_net = HyperNet(
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HYPERNET_HIDDEN_DIM,
        total_params_target=total_target_params
    ).to(config.DEVICE)
    
    z = torch.randn(1, config.LATENT_DIM, requires_grad=True, device=config.DEVICE)

    optimizer = optim.Adam(list(hyper_net.parameters()) + [z], lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Model Summary:")
    print(f"Target Network ({type(target_net_template).__name__}) requires {total_target_params:,} parameters.")
    hyper_net_params = sum(p.numel() for p in hyper_net.parameters())
    print(f"Hypernetwork ({type(hyper_net).__name__}) has {hyper_net_params:,} parameters.")
    print(f"Latent vector 'z' has {z.numel()} parameters.")
    print(f"Total trainable parameters: {hyper_net_params + z.numel():,}")

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(config.EPOCHS):
        hyper_net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()

            generated_weights_vector = hyper_net(z).squeeze(0)
            target_weights = vector_to_target_weights(generated_weights_vector, param_dict)

            outputs = target_net_template.forward(images, target_weights)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        hyper_net.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():

            generated_weights_vector = hyper_net(z).squeeze(0)
            target_weights = vector_to_target_weights(generated_weights_vector, param_dict)

            for images, labels in test_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = target_net_template.forward(images, target_weights)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print("\nTraining Finished")
    plot_training_results(train_losses, train_accs, test_accs)


if __name__ == '__main__':
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/__init__.py', 'w') as f:
        pass
        
    main()
