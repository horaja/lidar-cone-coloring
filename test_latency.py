import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import time  # Import the time module
from train import LiDARColorCNN, NpyConeDataset # Reuse components
import matplotlib.pyplot as plt

# ===================================================================
# Main Execution Block for Testing
# ===================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test a trained CNN on real LiDAR cone data.')
  parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the test dataset.')
  parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth file).')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"--- Running Inference on {device} ---")

  # --- Dataset and Dataloader ---
  test_dataset = NpyConeDataset(root=args.data_dir)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
  class_names = test_dataset.classes
  print(f"Found {len(test_dataset)} samples in {len(class_names)} classes: {class_names}")

  # --- Load Model ---
  model = LiDARColorCNN(num_classes=len(class_names)).to(device)
  model.load_state_dict(torch.load(args.model_path, map_location=device))
  model.eval()

  # --- Inference Loop & Performance Measurement ---
  correct = 0
  total = 0
  misclassified_samples = []
  
  # Start the timer right before the loop
  start_time = time.time()

  with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Store misclassified samples for visualization
      for j in range(images.size(0)):
        if predicted[j] != labels[j]:
          misclassified_samples.append({
            'image': images[j].cpu().numpy().squeeze(),
            'predicted': class_names[predicted[j]],
            'actual': class_names[labels[j]]
          })
          
  # Stop the timer after the loop
  end_time = time.time()
  total_inference_time = end_time - start_time

  # --- Calculate Metrics ---
  num_samples = len(test_dataset)
  num_batches = len(test_loader)
  
  # Latency: The average time to process a single item or batch
  avg_latency_batch = (total_inference_time / num_batches) * 1000  # in milliseconds
  avg_latency_sample = (total_inference_time / num_samples) * 1000 # in milliseconds

  # Throughput: The number of samples processed per second
  throughput_fps = num_samples / total_inference_time

  # --- Report Results ---
  accuracy = 100 * correct / total
  print(f"\n--- Testing Finished ---")
  
  # --- Accuracy Metrics ---
  print("\n--- Accuracy Metrics ---")
  print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")

  # --- Performance Metrics ---
  print("\n--- Performance Metrics ---")
  print(f"  Total Inference Time: {total_inference_time:.3f} seconds")
  print(f"  Average Latency per Batch: {avg_latency_batch:.3f} ms")
  print(f"  Average Latency per Sample: {avg_latency_sample:.3f} ms")
  print(f"  Throughput: {throughput_fps:.2f} samples/sec (FPS)")


  # --- Visualize Misclassified Samples ---
  if misclassified_samples:
    print(f"\nVisualizing up to 10 misclassified samples...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, sample in enumerate(misclassified_samples[:10]):
      ax = axes[i]
      ax.imshow(sample['image'], cmap='gray')
      ax.set_title(f"Pred: {sample['predicted']}\nActual: {sample['actual']}")
      ax.axis('off')
    plt.tight_layout()
    # Save the figure instead of showing it, for cluster environments
    plt.savefig('misclassified_samples.png')
    print("Saved misclassified samples plot to misclassified_samples.png")