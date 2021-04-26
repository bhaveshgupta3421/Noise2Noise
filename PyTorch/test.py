import matplotlib.pyplot as plt
import torch
from train import dataloaders
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device.type))

def testing(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)