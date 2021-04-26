import torch
import time
import copy
from torch.utils.data import DataLoader
from data_load import LoadImageDataset, resize_noise_image
import matplotlib.pyplot as plt
from network import UNet
from torch.optim import lr_scheduler
from torch import nn
import torch.optim as optim

train_dir = "../dataset/train"
val_dir = "../dataset/test"

train_data = LoadImageDataset(train_dir, resize_noise_image)
val_data = LoadImageDataset(val_dir, resize_noise_image)

dataset_sizes = {
    'train': train_data.__len__(),
    'val': val_data.__len__()
    }

dataloaders = {
    'train': DataLoader(train_data, batch_size=64, shuffle=True),
    'val': DataLoader(val_data, batch_size=64, shuffle=True)
    }

#####################        Visualizing Dataset         #####################

noised_inp, noised_mask = next(iter(dataloaders['train']))
print(f"Feature batch shape: {noised_inp.size()}")
print(f"Labels batch shape: {noised_mask.size()}")

img = noised_inp[0].squeeze()
label = noised_mask[0].squeeze()
plt.imshow(label)
plt.show()

plt.imshow(img)
plt.show()
##############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device.type))

def training(model, loss_func, optimizer, scheduler, num_epochs=20):
    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch No. --> {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                epoch_loss = best_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model

unet_model = UNet(1,2)

input_image = torch.rand(1,1,256,256)
final_output = unet_model(input_image)
print(final_output.size())
for name, param in unet_model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} \n")
    
unet_model = unet_model.to(device)

loss_func = nn.MSELoss()

optimizer_ft = optim.Adam(unet_model.parameters(), lr=0.003, betas=(0.9, 0.99), eps=1e-08)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

unet_model = training(unet_model, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=1)