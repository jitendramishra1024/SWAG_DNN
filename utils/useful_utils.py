def modify_train_loss_accuracy_from_batch_level_to_epoch_level(train_acc,train_losses):
  new_train_acc=[]
  sum=0
  count=0
  for i in range (len(train_acc)):
    sum=sum+train_acc[i]
    count=count+1
    if count%len(trainloader)==0:
      new_train_acc.append(sumlen(trainloader))
      sum=0


  new_train_loss=[]
  sum=0
  count=0
  for i in range (len(train_losses)):
    sum=sum+train_losses[i]
    count=count+1
    if count%len(trainloader)==0:
      new_train_loss.append(sumlen(trainloader))
      sum=0
  return new_train_acc,new_train_loss