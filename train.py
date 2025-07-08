import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import os
import argparse

# ===================================================================
# 1. Data Simulator
# ===================================================================
class ConeSimulator:
  """
  Generates realistic 32x32 cone intensity grids for training a CNN.
  Produces patterns for blue cones, yellow cones, and unknown/noisy signals.
  """
  def __init__(self, grid_size=(32, 32)):
    self.grid_size = grid_size
    self.labels = {'blue': 0, 'yellow': 1, 'unknown': 2}

  def _create_base_grid(self):
    return np.zeros(self.grid_size, dtype=np.float32)

  def _generate_cone_pattern(self, intensity_bands):
    """
    Generates a base cone pattern based on intensity bands.
    Args:
        intensity_bands (dict): Keys are (start_row, end_row) and 
                                values are (min_intensity, max_intensity, num_lines).
    """
    grid = self._create_base_grid()
    grid_height, grid_width = self.grid_size

    for (start_row, end_row), (min_int, max_int, num_lines) in intensity_bands.items():
      if end_row > start_row:
        active_rows = np.random.choice(range(start_row, end_row + 1), size=num_lines, replace=True)
      else:
        active_rows = [start_row]

      for r in active_rows:
        cone_half_width = int((grid_width / 2) * (r / grid_height) * 0.9 + 3)
        start_col = max(0, grid_width // 2 - cone_half_width)
        end_col = min(grid_width, grid_width // 2 + cone_half_width)

        if end_col > start_col:
          points_in_line = np.random.randint(cone_half_width, int(cone_half_width * 1.5))
          cols = np.random.randint(start_col, end_col, size=points_in_line)
          grid[r, cols] += np.random.uniform(min_int, max_int, size=points_in_line)

    return grid

  def generate_blue_cone(self):
    """Generates blue cone: low-intensity top/bottom, high-intensity middle."""
    bands = {
      (2, 10): (10.0, 25.0, np.random.randint(3, 7)),
      (11, 23): (35.0, 70.0, np.random.randint(7, 12)),
      (24, 30): (10.0, 25.0, np.random.randint(2, 5))
    }
    return self._generate_cone_pattern(bands)

  def generate_yellow_cone(self):
    """Generates yellow cone: high-intensity top/bottom, low-intensity middle."""
    bands = {
      (2, 10): (35.0, 70.0, np.random.randint(3, 6)),
      (11, 23): (10.0, 25.0, np.random.randint(5, 10)),
      (24, 30): (35.0, 70.0, np.random.randint(3, 6))
    }
    return self._generate_cone_pattern(bands)

  def generate_unknown(self, num_points_range=(10, 60)):
    """Generates noisy signal that doesn't form a clear cone (unknown)."""
    unknown_grid = self._create_base_grid()
    num_points = np.random.randint(num_points_range[0], num_points_range[1])
    rows = np.random.randint(0, self.grid_size[0], size=num_points)
    cols = np.random.randint(0, self.grid_size[1], size=num_points)
    unknown_grid[rows, cols] = np.random.uniform(20.0, 100.0, size=num_points)
    return unknown_grid

  def generate_sample(self):
    """
    Generates a single augmented sample with random rotation, zoom, translation, noise, and jitter.
    """
    category = np.random.choice(list(self.labels.keys()))
    label = self.labels[category]

    if category == 'blue':
      image = self.generate_blue_cone()
    elif category == 'yellow':
      image = self.generate_yellow_cone()
    else:
      image = self.generate_unknown()

    h, w = image.shape
    angle = np.random.uniform(-15, 15)
    scale = np.random.uniform(0.4, 1.1)
    tx = np.random.uniform(-2, 2)
    ty = np.random.uniform(-2, 2)

    center = (w // 2, h // 2)
    trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    trans_matrix[0, 2] += tx
    trans_matrix[1, 2] += ty

    image = cv2.warpAffine(image, trans_matrix, (w, h), borderValue=0)
    image = gaussian_filter(image, sigma=(0.4, 1.2))
    noise = np.random.uniform(0, 5.0, self.grid_size)
    image += noise

    image = np.clip(image, 0, None)
    min_val, max_val = np.min(image), np.max(image)
    if max_val > min_val:
      image = (image - min_val) / (max_val - min_val)

    return image.astype(np.float32), label

# ===================================================================
# 2. PyTorch Dataset
# ===================================================================
class ConeDataset(Dataset):
  """Generates cone data on-the-fly."""
  def __init__(self, num_samples=1000):
    self.num_samples = num_samples
    self.simulator = ConeSimulator()

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    image, label = self.simulator.generate_sample()
    image = torch.from_numpy(image).unsqueeze(0) # Adds channel dimension -- why?
    return image, torch.tensor(label, dtype=torch.long)

# ===================================================================
# 3. Custom Loss Function
# ===================================================================
class ConePenaltyLoss(nn.Module):
  """Applies a heavy penalty for confusing blue and yellow cones."""
  def __init__(self, heavy_penalty):
    super(ConePenaltyLoss, self).__init__()
    self.heavy_penalty = heavy_penalty
    self.base_loss = nn.CrossEntropyLoss(reduction='none')

  def forward(self, logits, targets):
    base_loss = self.base_loss(logits, targets)
    preds = torch.argmax(logits, dim=1)
    penalties = torch.ones_like(targets, dtype=torch.float)

    # Critical errors: predicting blue as yellow or yellow as blue
    blue_as_yellow = (preds == 0) & (targets == 1)
    yellow_as_blue = (preds == 1) & (targets == 0)
    critical_errors = blue_as_yellow | yellow_as_blue

    penalties[critical_errors] = self.heavy_penalty
    return (base_loss * penalties).mean()

# ===================================================================
# 4. CNN Model Architecture
# ===================================================================
class LiDARColorCNN(nn.Module):
  """CNN for classifying cone colors from simulated LiDAR data."""
  def __init__(self, num_classes=3):
    super(LiDARColorCNN, self).__init__()
    self.feature_extractor = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Flatten()
    )
    self.classifier = nn.Sequential(
      nn.Linear(256 * 2 * 2, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(128, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(64, num_classes),
    )
  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x

# ===================================================================
# 5. Main Execution Block
# ===================================================================
if __name__ == '__main__':
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser(description='Train a CNN for LiDAR Cone Coloring.')
  parser.add_argument('--num_classes', type=int, default=3, help='Number of classes.')
  parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs.')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--num_train_samples', type=int, default=5000, help='Number of training samples to generate per epoch.')
  parser.add_argument('--num_val_samples', type=int, default=30, help='Number of validation samples to test.')
  parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and validation images.')
  parser.add_argument('--model_save_path', type=str, default='models/cone_classifier.pth', help='Path to save the trained model.')

  args = parser.parse_args()

  # --- Setup ---
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  os.makedirs(args.log_dir, exist_ok=True)
  os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

  train_dataset = ConeDataset(num_samples=args.num_train_samples)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

  model = LiDARColorCNN(num_classes=args.num_classes).to(device)
  criterion = ConePenaltyLoss(heavy_penalty=5.0)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  print(f'--- Starting Training on {device} ---')
  print(f'Hyperparameters: {vars(args)}')

  # --- Training Loop ---
  for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

  print("--- Training Finished ---")
  
  # --- Save Model ---
  torch.save(model.state_dict(), args.model_save_path)
  print(f"Model saved to {args.model_save_path}")

  # --- Validation Loop ---
  print(f"\\n--- Running Validation on {args.num_val_samples} New Samples ---")
  model.eval()
  simulator = ConeSimulator()
  class_names = {v: k for k, v in simulator.labels.items()}
  correct_predictions = 0

  with torch.no_grad():
    for i in range(args.num_val_samples):
      sample_image_np, actual_label = simulator.generate_sample()
      sample_image_tensor = torch.from_numpy(sample_image_np).unsqueeze(0).unsqueeze(0).to(device)
      
      pred = model(sample_image_tensor)
      prob = torch.softmax(pred, dim=1)
      pred_class_idx = torch.argmax(prob, dim=1).item()
      
      if pred_class_idx == actual_label:
          correct_predictions += 1

      predicted_name = class_names[pred_class_idx].title()
      actual_name = class_names[actual_label].title()
      
      print(f"\\n--- Sample {i+1}/{args.num_val_samples} ---")
      print(f"Actual Class: '{actual_name}' (Label: {actual_label})")
      print(f"Predicted Class: '{predicted_name}' (Label: {pred_class_idx})")
      print(f"Confidence: {prob.max().item():.2%}")

      # Save the validation image to a file
      fig = plt.figure(figsize=(6, 6))
      plt.imshow(sample_image_np, cmap='gray', vmin=0, vmax=1)
      plt.title(f'Prediction: {predicted_name}\\n(Actual: {actual_name})', fontsize=14)
      plt.axis('off')
      
      image_path = os.path.join(args.log_dir, f'validation_sample_{i+1}.png')
      plt.savefig(image_path)
      plt.close(fig) # Close the figure to free memory

  accuracy = (correct_predictions / args.num_val_samples) * 100
  print(f"\\n--- Validation Finished ---")
  print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{args.num_val_samples})")