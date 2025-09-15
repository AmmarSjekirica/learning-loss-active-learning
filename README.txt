This repository demonstrates a basic implementation of "Learning Loss for Active Learning," 
where a neural network (for classification) and an auxiliary loss prediction module are trained 
jointly. The code uses PyTorch and downloads the CIFAR-10 dataset automatically.

## Contents

- **main.py**: Main script to orchestrate data loading, model initialization, training,
  and active learning cycles.
- **model.py**: Defines the `SimpleCNN` class, a CNN for image classification with a 
  secondary head for loss prediction.
- **active_learning.py**: Contains the function to select unlabeled samples based on 
  predicted loss.
