import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os

from Hypernetworks_stevens import PrimaryNetwork
from Hypernetworks_stevens import Battery_dataset


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model, loss, optimizer
model = PrimaryNetwork()
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)  
optimizer = optim.Adam(model.parameters(), lr=1e-3)

image_dir = "/home/CAMPUS/sgangadh1/projects/rl-batt-seg-snapshot-jan-2024/data/battery_2/train_images"
label_dir = "/home/CAMPUS/sgangadh1/projects/rl-batt-seg-snapshot-jan-2024/data/battery_2/train_label"

# Load dataset
train_dataset = Battery_dataset(image_dir, label_dir)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# Training loop
num_epochs = 100
start_epoch = 0
end_epoch = 100
num_batches = 16

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

checkpoint_path = "mar27_exp_battery/checkpoint_epoch_0.pth"  

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0  # Start from scratch if no checkpoint found
    print("No checkpoint found. Starting from epoch 0.")

# Initialize lists to store true and predicted labels
# Training loop
total_true_labels = []
total_preds = []
loss_per_epoch = []
losses = []

# Training loop
for epoch in range(start_epoch, end_epoch+1):
    model.train()
    loss_epoch = 0
    loss_per_batch = []
    all_preds = []
    all_true_labels = []
    num_left_count = 0  # Reset for each epoch
    data_len = 0  # Track the number of valid batches

    for all_patches, x_hyp, key_pixels, all_labels in train_loader:
        for batch in range(len(all_patches)):
            batch_patches = all_patches[batch].to(device)
            batch_pixels = key_pixels[batch].to(device)
            batch_labels = all_labels[batch].to(device)
            batch_mask_img = x_hyp[batch].to(device)
            batch_loss = 0
            count = 0  # Count of skipped pixels

            for i, pixel in enumerate(batch_pixels):
                if (batch_labels[i] != 255).sum() == 0:
                    num_left_count += 1
                    count += 1
                    continue
                
                optimizer.zero_grad()
                output = model(batch_patches[i], batch_mask_img, pixel)
                loss = criterion(output, batch_labels[i])
                batch_loss += loss

                # Get predictions
                preds = output.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().flatten())
                total_preds.extend(preds.cpu().numpy().flatten())
                total_true_labels.extend(batch_labels[i].cpu().numpy().flatten())
                all_true_labels.extend(batch_labels[i].cpu().numpy().flatten())

                loss.backward()
                optimizer.step()

            # Normalize batch_loss by valid pixels
            if len(batch_pixels) - count > 0:
                batch_loss /= (len(batch_pixels) - count)
                loss_per_batch.append(batch_loss.item())  # Storeing batch loss only if valid
                loss_epoch += batch_loss.item()
                data_len += 1  # Increment count of valid batches

    # Compute metrics
    epoch_loss = loss_epoch / data_len if data_len > 0 else 0  # Avoid division by zero
    loss_per_epoch.append(epoch_loss)

    accuracy = accuracy_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten())
    f1_micro = f1_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten(), average='micro')
    f1_macro = f1_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten(), average='macro')
    precision = precision_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten(), average='macro', zero_division=0)
    recall = recall_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten(), average='macro', zero_division=0)
    iou = jaccard_score(np.array(all_true_labels).flatten(), np.array(all_preds).flatten(), average='macro')

    # Logging
    log_file = "mar27_exp_battery/training_log.txt"
    with open(log_file, "a") as f:
        log_message = (f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
                       f"Accuracy: {accuracy:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}\n"
                       f"Number of patches with no valid pixels: {num_left_count}\n\n")
        print(log_message, end="")  # Print to console
        f.write(log_message)  # Write to file

    # Save checkpoint
    # Save checkpoint only every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss  # Store averaged loss
        }
    torch.save(checkpoint, f'mar27_exp_battery/checkpoint_epoch_{epoch+1}.pth')
    print(f"Checkpoint saved at epoch {epoch+1}")


# Save losses
import pickle
with open("mar27_exp_battery/loss_per_epoch_100.pkl", "wb") as file:
    pickle.dump(loss_per_epoch, file)
with open("mar27_exp_battery/losses_100.pkl", "wb") as file:
    pickle.dump(losses, file)
with open("mar27_exp_battery/total_true_labels_100.pkl", "wb") as file:
    pickle.dump(total_true_labels, file)
with open("mar27_exp_battery/total_preds_100.pkl", "wb") as file:
    pickle.dump(total_preds, file)
