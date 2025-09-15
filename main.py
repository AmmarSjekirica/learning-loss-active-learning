import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from model import SimpleCNN
from active_learning import select_samples_by_loss


def train_one_epoch(model, loader, optimizer, device='cpu'):
    """
    Trains the model for one epoch on the given DataLoader.
    Combines classification loss and loss prediction loss.
    """
    model.train()
    classification_criterion = nn.CrossEntropyLoss(reduction='none')
    loss_prediction_criterion = nn.L1Loss()

    total_combined_loss = 0.0
    total_cls_loss = 0.0
    total_lp_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, loss_pred = model(images)

        # Sample-wise cross-entropy
        cls_losses = classification_criterion(logits, labels)  # shape: [batch_size]
        cls_loss = cls_losses.mean()  # average across batch

        # The model's predicted loss should match the actual cross-entropy
        lp_loss = loss_prediction_criterion(loss_pred, cls_losses.detach())

        combined_loss = cls_loss + lp_loss
        combined_loss.backward()
        optimizer.step()

        total_combined_loss += combined_loss.item()
        total_cls_loss += cls_loss.item()
        total_lp_loss += lp_loss.item()

    n_batches = len(loader)
    return {
        'combined': total_combined_loss / n_batches,
        'cls_loss': total_cls_loss / n_batches,
        'lp_loss': total_lp_loss / n_batches
    }


def evaluate(model, loader, device='cpu'):
    """
    Evaluates classification accuracy on a given dataset loader (e.g., test set).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, loss_pred = model(images)
            _, predicted = logits.max(dim=1)  # predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    # Hyperparameters
    initial_labeled_size = 1000  # number of samples initially labeled
    query_size = 100  # how many new samples to query each cycle
    cycles = 5  # number of active learning cycles
    epochs_per_cycle = 5  # train for this many epochs each cycle
    batch_size = 64
    lr = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1) CIFAR-10 data, auto-download
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 2) Split into labeled/unlabeled
    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_labeled_size]
    unlabeled_indices = all_indices[initial_labeled_size:]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3) Create model, optimizer
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) Active learning loop
    for cycle in range(cycles):
        print(f"\n=== Active Learning Cycle {cycle + 1}/{cycles} ===")

        # Subsets for labeled/unlabeled
        labeled_subset = Subset(train_dataset, labeled_indices)
        unlabeled_subset = Subset(train_dataset, unlabeled_indices)

        labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)

        # Train
        for epoch in range(epochs_per_cycle):
            metrics = train_one_epoch(model, labeled_loader, optimizer, device)
            print(f"  Epoch {epoch + 1}/{epochs_per_cycle} | "
                  f"Combined Loss: {metrics['combined']:.4f}, "
                  f"Cls Loss: {metrics['cls_loss']:.4f}, "
                  f"LP Loss: {metrics['lp_loss']:.4f}")

        # Evaluate on test set
        acc = evaluate(model, test_loader, device)
        print(f"  Test Accuracy after cycle {cycle + 1}: {acc * 100:.2f}%")

        # Select top-K by predicted loss
        newly_selected = select_samples_by_loss(model, unlabeled_loader, device=device, K=query_size)

        # Update labeled/unlabeled pools
        labeled_indices.extend(newly_selected)
        unlabeled_indices = list(set(unlabeled_indices) - set(newly_selected))

    # Final evaluation
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
