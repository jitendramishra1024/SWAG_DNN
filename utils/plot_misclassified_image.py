import math
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_misclassified(model, test_loader, device):
  misclassified = []
  misclassified_pred = []
  misclassified_target = []
  # put the model to evaluation mode
  model.eval()
  # turn off gradients
  with torch.no_grad():
      for data, target in test_loader:
        # move them to the respective device
        data, target = data.to(device), target.to(device)
        # do inferencing
        output = model(data)
        # get the predicted output
        pred = output.argmax(dim=1, keepdim=True)

        # get the current misclassified in this batch
        list_misclassified = (pred.eq(target.view_as(pred)) == False).squeeze()
        #list_misclassified  = [item for sublist in list_misclassified  for item in sublist]
        #print(list_misclassified)
        #print(data.shape)
        batch_misclassified = data[list_misclassified]
        batch_mis_pred = pred[list_misclassified]
        batch_mis_target = target.view_as(pred)[list_misclassified]

        misclassified.append(batch_misclassified)
        misclassified_pred.append(batch_mis_pred)
        misclassified_target.append(batch_mis_target)
        #print(len(misclassified))

  # group all the batched together
  misclassified = torch.cat(misclassified)
  misclassified_pred = torch.cat(misclassified_pred)
  misclassified_target = torch.cat(misclassified_target)

  return  misclassified, misclassified_pred, misclassified_target
  
def get_properclassified(model, test_loader, device):
    classified = []
    misclassified_pred = []
    misclassified_target = []
    # put the model to evaluation mode
    model.eval()
    # turn off gradients
    with torch.no_grad():
      for data, target in test_loader:
        # move them to the respective device
        data, target = data.to(device), target.to(device)
        # do inferencing
        output = model(data)
        # get the predicted output
        pred = output.argmax(dim=1, keepdim=True)

        # get the current misclassified in this batch
        list_misclassified = (pred.eq(target.view_as(pred)) == True).squeeze()
        #list_misclassified  = [item for sublist in list_misclassified  for item in sublist]
        #print(list_misclassified)
        #print(data.shape)
        batch_misclassified = data[list_misclassified]
        batch_mis_pred = pred[list_misclassified]
        batch_mis_target = target.view_as(pred)[list_misclassified]

        classified.append(batch_misclassified)
        misclassified_pred.append(batch_mis_pred)
        misclassified_target.append(batch_mis_target)
        #print(len(misclassified))

    # group all the batched together
    classified = torch.cat(misclassified)
    misclassified_pred = torch.cat(misclassified_pred)
    misclassified_target = torch.cat(misclassified_target)

  return  classified, misclassified_pred, misclassified_target
  
  
  
 
def plot_misclassified(number,test_loader, device,model,classes,mean,std,format ):
    images, predicted, actual = get_misclassified(model,test_loader,device)
    nrows = math.floor(math.sqrt(number))
    ncols = math.ceil(math.sqrt(number))

    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title("Predicted: %s\nActual: %s" % (classes[predicted[index]], classes[actual[index]]))
            #FIRST UNNORMALIZE THEN SHOW 
            mean = np.array(mean)
            std = np.array(std)
            if format=='raw':
              ax[i, j].imshow(np.transpose(images[index].cpu().numpy(), (1, 2, 0))*std+mean, cmap="gray_r")
            elif format=='normalized':
              ax[i, j].imshow(np.transpose(images[index].cpu().numpy(), (1, 2, 0)), cmap="gray_r")