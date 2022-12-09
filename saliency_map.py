from torchvision import transforms
from PIL import Image

import argparse 

# PyTorch libraries 
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torchvision.datasets import DatasetFolder
from torchvision import models, transforms 
from transformers import ViTFeatureExtractor, ViTForImageClassification
from models import BasicCNNModel

import matplotlib.pyplot as plt

def get_saliency_map(model_param_path, image_path):
    #model = torch.load(model_param_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicCNNModel(3)
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    model.eval()

    image = np.load(image_path) # get image from dataset
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # transforms_ = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),  
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # img = transforms_(image)
    # img = img.reshape(1, 3, 224, 224)

    image = feature_extractor(image, return_tensors='pt')['pixel_values']
    image = image.to(device)
    print(type(image))
    image.requires_grad_()
    output = model(image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    image = image.reshape(-1, 224, 224)

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('Image and Saliency Map')
    plt.savefig('img.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # example args would be 
    # main.py --data_dir west_africa_npy --csv_file west_africa_df --outcome Mean_BMI_bin  --n_classes 3
    parser.add_argument('--model_param_path', type=str, help='Image data location.')
    parser.add_argument('--image_path', type=str, help='Image data location.')
    args = parser.parse_args()

    get_saliency_map(args.model_param_path, args.image_path)