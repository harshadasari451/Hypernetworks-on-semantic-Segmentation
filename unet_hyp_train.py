import torch
from torch.utils.data import DataLoader
from src.models.Unet_model import UNet, UNetTinyTwoBlock
from src.data_loaders.Battery_unet_hyp_dataloader import Battery_unet_hyp_data
from src.models.primaryNet import PrimaryNetwork
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from torchmetrics.functional import jaccard_index




device_1 = torch.device("cpu")
print(f"Using device for unet and unet_tiny models : {device_1}")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

unet_path = "/home/CAMPUS/sgangadh1/projects/rl-batt-seg-snapshot-jan-2024/src/outputs/rerun-battery-01/unet_model_checkpoint_finetuned.pt"
global_model_path = "/home/CAMPUS/hdasari/HyperNetworks/model_checkpoints_AAI_paper/unet_tiny_two_block_model_checkpoint_199_final.pt"

expert_model_unet = UNet(n_channels=1, n_classes=3)
expert_model_unet = expert_model_unet.to(device_1)

global_model = UNetTinyTwoBlock(n_channels=1, n_classes=3)
global_model = global_model.to(device_1)


expert_checkpoint = torch.load(unet_path)
global_checkpoint = torch.load(global_model_path)

expert_model_unet.load_state_dict(expert_checkpoint['model_state_dict'])
expert_model_unet.eval()

global_model.load_state_dict(global_checkpoint['model_state_dict'])
global_model.eval()


image_dir = "/home/CAMPUS/sgangadh1/projects/rl-batt-seg-snapshot-jan-2024/data/battery_2/train_images"
label_dir = "/home/CAMPUS/sgangadh1/projects/rl-batt-seg-snapshot-jan-2024/data/battery_2/train_label"


# Load dataset
train_dataset = Battery_unet_hyp_data(image_dir, label_dir, expert_model_unet, global_model,device_1)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)


model = PrimaryNetwork()
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)  
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
start_epoch = 0
end_epoch = 100
num_batches = 1

checkpoint_path = "/home/CAMPUS/hdasari/HyperNetworks/unet_hyp_apr9/checkpoint_epoch_0.pth"  

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
for epoch in range(start_epoch, end_epoch):
    model.train()
    loss_epoch = 0
    loss_per_batch = []
    all_preds = []
    all_true_labels = []
    num_left_count = 0  # Reset for each epoch
    data_len = 0  # Track the number of valid batches

    for all_patches, global_patches, expert_patches, all_labels,mismatch in train_loader:

        for batch in range(len(all_patches)):
            batch_patches = all_patches[batch].to(device)
            batch_global_patches = global_patches[batch].to(device)
            batch_expert_patches = expert_patches[batch].to(device)
            batch_labels = all_labels[batch].to(device)
            mismatch_count = mismatch[batch]

            
            batch_loss = 0
            count = 0  # Count of skipped pixels

            for patch in range(len(batch_patches)):

                optimizer.zero_grad()
                output = model(batch_patches[patch],batch_expert_patches[patch],batch_global_patches[patch])
                loss = criterion(output, batch_labels[patch])

                batch_loss += loss

                # Get predictions
                preds = output.argmax(dim=-1)
                preds_np = preds.cpu().numpy().flatten()
                true_np = batch_labels[patch].cpu().numpy().flatten()

                # Only keep entries where the ground truth is not 255
                valid_mask = true_np != 255

                # Filter predictions and ground truths
                filtered_preds = preds_np[valid_mask]
                filtered_true = true_np[valid_mask]

                # Append the filtered results
                all_preds.extend(filtered_preds)
                total_preds.extend(filtered_preds)
                all_true_labels.extend(filtered_true)
                total_true_labels.extend(filtered_true)

                loss.backward()
                optimizer.step()

            # Normalize batch_loss by valid pixels
            if len(batch_expert_patches) - count > 0:
                batch_loss /= (len(batch_expert_patches) - count)
                loss_per_batch.append(batch_loss.item())  # Storing batch loss only if valid
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
    iou = jaccard_index(torch.tensor(all_preds),torch.tensor(all_true_labels), task="multiclass", num_classes=3)

    # Logging
    log_file = "exp_apr10/training_log.txt"
    with open(log_file, "a") as f:
        log_message = (f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
                       f"Accuracy: {accuracy:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}\n"
                       f"Number of patches with no valid pixels: {num_left_count}\n\n")
        # print(log_message, end="")  # Print to console
        f.write(log_message)  # Write to file

    # Save checkpoint
    # Save checkpoint only every 10 epochs
    if epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, accuracy: {accuracy:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, accuracy: {accuracy:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss  # Store averaged loss
        }
        torch.save(checkpoint, f'exp_apr10/checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")


# Save losses
import pickle
with open("exp_apr10/loss_per_epoch_100.pkl", "wb") as file:
    pickle.dump(loss_per_epoch, file)
with open("exp_apr10/losses_100.pkl", "wb") as file:
    pickle.dump(losses, file)
with open("exp_apr10/total_true_labels_100.pkl", "wb") as file:
    pickle.dump(total_true_labels, file)
with open("exp_apr10/total_preds_100.pkl", "wb") as file:
    pickle.dump(total_preds, file)