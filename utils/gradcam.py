import copy
from torchvision.utils import make_grid, save_image
import torchvision
import torchvision.transforms as transforms
import torch
#gradcam 
import torch.nn.functional as F
import PIL
class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, model, layer_name):
        self.model = model
        # self.layer_name = layer_name
        self.target_layer = layer_name

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        self.gradients.clear()
        self.activations.clear()
        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
        
        
        

import cv2
def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result










#helper function 
def show_one_row_graph(images,mean,std,pred,actual,format='raw',mode="result"):
  nrows=1
  ncols = 5
  l=['image','layer1','layer2','layer3','layer4']
  if mode=="result":
    l[0]="Result :P:"+pred+"A:"+actual
  else :
    l[0]="HeatMap :P:"+pred+"A:"+actual

  fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))
  for j in range(ncols):
      index =  j
      ax[j].axis("off")
      ax[j].set_title(l[j])
      #FIRST UNNORMALIZE THEN SHOW 
      mean = np.array(mean)
      std = np.array(std)
      if format=='raw':
        ax[j].imshow(np.transpose(images[index].cpu().numpy(), (1, 2, 0))*std+mean, cmap="gray_r")
      elif format=='normalized':
        ax[j].imshow(np.transpose(images[index].cpu().numpy(), (1, 2, 0)), cmap="gray_r")
def show_gradcam_for_n_images(n,images,predicted,actual,classes,model):
  torch_img_list=missed_images[0:n]
  pred_list=predicted[0:n]
  actual_list=actual[0:n]
  classes=classes
  torch_img_list=[x.reshape(3, 32, 32) for x in torch_img_list ]
  normed_torch_img=[x.reshape(1,3, 32, 32) for x in torch_img_list ]
  for i,k in enumerate(normed_torch_img):
    pred=classes[pred_list[i]]
    actual=classes[actual_list[i]]
    images1 = [torch_img_list[i].cpu()]
    #images1 = np.transpose(torch_img_list[i].cpu().numpy(), (1, 2, 0))*std+mean
    images2 =  [torch_img_list[i].cpu()]
    b = copy.deepcopy(model.to(device))
    layers =  [b.layer1,b.layer2,b.layer3,b.layer4]
    for j in layers:
      g = GradCAM(b,j)
      mask, _= g(normed_torch_img[i])
      heatmap, result = visualize_cam(mask,torch_img_list[i] )
      #heatmap, result = visualize_cam(mask,unnormalize(images1, mean, std, out_type='tensor').clone().unsqueeze_(0).to(self.device) )
      images1.extend([heatmap])
      images2.extend([result])
    show_one_row_graph(images1,mean,std,pred,actual,format='raw',mode="heatmap")
    show_one_row_graph(images2,mean,std,pred,actual,format='raw',mode="result")
    
    
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchvision.utils import make_grid, save_image
def imshow(img,std,mean):
    #img = img / 2 + 0.5     # unnormalize
    mean = np.array(mean)
    std = np.array(std)
    npimg=np.transpose(img.cpu().numpy(), (1, 2, 0))*std+mean
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')

def grad_cam_for_user_defined_images(img_name,device,mean,std):
  pil_image = []
  for i,img in enumerate(img_name):
    pil_image.append(PIL.Image.open(img))
  normed_torch_img = []
  torch_img_list = []

  for i in pil_image:
    torch_img = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])(i).to(device)
    torch_img_list.append(torch_img)

    normed_torch_img .append(transforms.Normalize(mean, std)(torch_img)[None])
  #import model
  for i,k in enumerate(normed_torch_img):
    images1 = [torch_img_list[i].cpu()]
    #images1 = np.transpose(torch_img_list[i].cpu().numpy(), (1, 2, 0))*std+mean
    images2 =  [torch_img_list[i].cpu()]
    b = copy.deepcopy(model.to(device))
    layers =  [b.layer1,b.layer2,b.layer3,b.layer4]
    for j in layers:
      g = GradCAM(b,j)
      mask, _= g(normed_torch_img[i])
      heatmap, result = visualize_cam(mask,torch_img_list[i] )
      images1.extend([heatmap])
      images2.extend([result])

    grid_image = make_grid(images1+images2, nrow=5)
    imshow(grid_image,std,mean)