from torchvision import transforms
from PIL import Image

# PyTorch libraries 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torchvision.datasets import DatasetFolder
from torchvision import models, transforms 

def get_saliency_map(model_param_path, image_path):
    model = torch.load(model_param_path)

    model.eval()

    image = Image.open(image_path) # get image from dataset

    transforms_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = transforms_(image)
    img = img.reshape(1, 3, 224, 224)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    image.requires_grad_()
    output = model(image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    img = img.reshape(-1, 224, 224)

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('Image and Saliency Map')
    plt.show()

if __name__ == '__main__':
    model_param_path = ''
    image_path = ''
    get_saliency_map(model_param_path, image_path)