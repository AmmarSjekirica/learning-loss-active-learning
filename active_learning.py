import torch


def select_samples_by_loss(model, unlabeled_loader, device='cpu', K=100):
    """
    Ranks unlabeled samples by predicted loss and returns the indices of the top K samples.

    :param model: The trained model (SimpleCNN) with a loss prediction head.
    :param unlabeled_loader: DataLoader for the unlabeled subset
    :param device: 'cpu' or 'cuda'
    :param K: number of samples to query
    :return: a list of global indices for the top K samples (descending predicted loss).
    """
    model.eval()
    predicted_losses = []
    sample_indices = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(unlabeled_loader):
            images = images.to(device)
            logits, loss_pred = model(images)

            # unlabeled_loader.dataset is a Subset with a .indices attribute
            # needed to map local batch indices to global dataset indices
            batch_global_indices = unlabeled_loader.dataset.indices

            for i in range(images.size(0)):
                sample_loss = loss_pred[i].item()
                global_idx = batch_global_indices[batch_idx * unlabeled_loader.batch_size + i]

                predicted_losses.append(sample_loss)
                sample_indices.append(global_idx)

    # Sort by predicted loss in descending order
    ranked = sorted(zip(sample_indices, predicted_losses), key=lambda x: x[1], reverse=True)

    # Take the top K
    selected_indices = [idx for idx, loss_val in ranked[:K]]
    return selected_indices
