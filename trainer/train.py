from tqdm import tqdm
import torch

class Trainer():

  def __init__(self):
    self.train_losses = []
    self.train_acc = []
      

  def train(self, model, device, train_loader, optimizer, loss_func, epoch, lambda_l1,scheduler=False ):
    #to find loss for each epoch for reduce LR on plateau
    epoch_loss_list=[]
    epoch_loss=1
    #if it is  step LR then update at every epoch      
    if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      loss = loss_func(y_pred, target)
      # L2 loss

      # L1 loss
      l1 = 0
      # lambda_l1 = 0.05
      for p in model.parameters():
        l1 = l1 + p.abs().sum()
      loss = loss + lambda_l1*l1
      #to append loss for each iteration 
      epoch_loss_list.append(loss)

      self.train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()
      #scheduler only in case of one cycle LR
      #if it is onecycle lr then update at every iteration 
      if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()


      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Train set: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      self.train_acc.append(100*correct/processed)
      tqdm._instances.clear()
    #if it is  reduce LR on plateau  then update at every epoch 

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if len(epoch_loss_list)!=0:
            epoch_loss=sum(epoch_loss_list)/len(epoch_loss_list)
            scheduler.step(epoch_loss)
            print("Epoch avarage loss for epoch "+str(epoch)+" is "+ str(epoch_loss))
        else :
            print("No loss is generated for any iteration  for this epoch")
        



  def getValuesTrain(self):
    return (self.train_losses, self.train_acc)