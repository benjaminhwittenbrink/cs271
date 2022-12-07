# standard libraries 

import argparse 
import os 
import random 
import pandas as pd
import numpy as np 
import collections 
from tqdm import tqdm 
from datetime import datetime

# PyTorch libraries 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision import models, transforms 

# Hugging Face datasets 
#import datasets 

# Transformers libraries 
from transformers import TrainingArguments, Trainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import get_linear_schedule_with_warmup 

# import sklearn models 
# from sklearn.neighbors import KNeighborsClassifier

# simple models
from models import LogisticRegression, BasicCNNModel, DenseCNNModel

from sklearn.metrics import confusion_matrix


RANDOM_SEED = 231 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class ImageClassificationCollator:
    def __init__(self, feature_extractor, transforms = False): 
        self.feature_extractor = feature_extractor
        self.transforms = transforms 

    def __call__(self, batch):
        if self.transforms: 
            transformed = [self.feature_extractor(x[0].cpu().detach().numpy()) for x in batch]
            encodings = {"pixel_values":torch.stack(transformed)}
        else: 
            encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')   
        encodings['labels'] = torch.tensor([x[1] for x in batch],  dtype=torch.long)
        return encodings

# create model and collator
def create_model_and_collator(args, model_name):

    if model_name == "ViT":
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        collator = ImageClassificationCollator(feature_extractor)
        collators = (collator, collator)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=CLASSES)

    elif model_name in ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet']:
        # note all models expect image of (3, 224, 224)

        train_data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224), # i.e. want 224 by 224 
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224), # i.e. want 224 by 224 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_collator = ImageClassificationCollator(train_data_transforms, transforms=True)
        val_collator = ImageClassificationCollator(val_data_transforms, transforms=True)

        collators = (train_collator, val_collator)

        if model_name == 'resnet':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, CLASSES)

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASSES)

        elif model_name == 'vgg':
            model = models.vgg11_bn(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASSES)

        elif model_name == 'squeezenet': 
            model = models.squeezenet1_0(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, CLASSES, kernel_size=1, stride=1)
            model.num_classes = CLASSES

        else: 
            # dense net 
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, CLASSES) 

    elif model_name in ['basic_cnn', 'dense_cnn', 'logistic_regression']:
        # ADD IN transforms though feature extractor might be easier 
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        collator = ImageClassificationCollator(feature_extractor)
        collators = (collator, collator)
        if model_name == "logistic_regression":
            model = LogisticRegression(n_classes=CLASSES)
        elif model_name == "basic_cnn":
            model = BasicCNNModel(n_classes=CLASSES)
        elif model_name == "dense_cnn":
            model = DenseCNNModel(n_classes=CLASSES)

    else: 
        raise NotImplementedError

    print(f'Model name: {model_name}')

    return collators, model 

class RegressionImageDataset(Dataset):

    def __init__(self, root, loader):

        self.root = root 
        self.image_paths = os.listdir(root)
        self.loader = loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_filepath = self.root + "/" + self.image_paths[idx]
        img = self.loader(image_filepath)

        label_string = self.image_paths[idx].split("_")[1].split(".")[0]
        label_string = label_string.replace("dot", ".")
        label = float(label_string)

        return img, label


def create_dataset(args, collator_fns, extensions = ['.npy'], val_split = 0.15):

    def npy_loader(path):
        sample = torch.from_numpy(np.load(path))
        return sample 
    
    # load in dataset frmom directory 
    dataset = RegressionImageDataset(
        root = args.data_dir, 
        loader = npy_loader
    )
    # split up into train val data  
    indices = torch.randperm(len(dataset)).tolist()
    n_val = int(np.floor(len(indices) * val_split))
    train_ds = torch.utils.data.Subset(dataset, indices[:-n_val])
    val_ds = torch.utils.data.Subset(dataset, indices[-n_val:])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator_fns[0], shuffle = 1)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator_fns[1], shuffle=0)

    return [train_dl, val_dl]


def validation(args, val_loader, model, criterion, device, name = 'Validation', write_file=None):

    model.eval()
    total_loss = 0. 
    # total_correct = 0 
    # total_sample = 0 
    # total_confusion = np.zeros((CLASSES, CLASSES))

    for i, batch in enumerate(tqdm(val_loader)):
        inputs, labels = batch['pixel_values'], batch['labels'] 
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if args.model_name in [
            'basic_cnn', 'dense_cnn', 'logistic_regression',
            'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
            ]:
                outputs = model(inputs)
            else: 
                outputs = model(inputs)['logits'] 

        labels = labels.float()
        labels = torch.unsqueeze(labels, 1)
        
        loss = criterion(outputs, labels)

        logits = outputs 
        total_loss += loss.cpu().item()


    print(f'*** Average batch loss on the {name} set: {total_loss / i})')
    cor = np.corrcoef(logits.cpu().numpy().squeeze(), labels.cpu().numpy().squeeze())
    print(f'*** Correlation between predictions and labels: {np.round(cor[0, 1], 2)}')
    print(f'*** Last five preds: {logits.cpu().numpy()[-5:]}')
    print(f'*** Last five labels: {labels.cpu().numpy()[-5:]}')

    return  total_loss



def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None):
    print("\n>>> Training starts...")

    if write_file: 
        write_file.write("\n>>> Training starts...")

    model.train()

    best_val_loss = 0
    for epoch in range(epoch_n):
        print("*** Epoch:", epoch)
        total_train_loss = 0. 

        for i, batch in enumerate(tqdm(data_loaders[0])): 
            optim.zero_grad()
            inputs, labels = batch['pixel_values'], batch['labels'] 
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward pass 
            if args.model_name in [
                'basic_cnn', 'dense_cnn', 'logistic_regression', 
                'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
            ]:
                outputs = model(inputs)
            else: 
                outputs = model(inputs)['logits']

            labels = labels.float()
            labels = torch.unsqueeze(labels, 1)
            
            loss = criterion(outputs, labels)
            logits = outputs

            total_train_loss += loss.cpu().item()

            # backward pass 
            loss.backward()
            optim.step()

            if scheduler: scheduler.step()

            if i % args.val_every == 0: 
                print(f'*** Average Loss: {total_train_loss / (i+1)}')
                if write_file:
                    write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')
                    write_file.write(f'*** Average Loss: {total_train_loss / (i+1)}')

                val_loss = validation(args, data_loaders[1], model, criterion, device, write_file=write_file)

                model.train()

                if best_val_loss < val_loss: 
                    best_val_loss = val_loss 

                    if args.save_path:
                        if args.model_name in ['ViT']:
                            model.save_pretrained(args.save_path)
                        else: 
                            torch.save(model.state_dict(), args.save_path)






if __name__ == '__main__':

    # set device to GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default = 'Mean_BMI_bin', help='Image data location.')

    #parser.add_argument('--n_classes', default=3, type=int, help='Number of classes in outcome variable.')	
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
    parser.add_argument('--epoch_n', default=10, type=int, help='Number of epochs for training.')
    parser.add_argument('--val_every', default=200, type=int, help="Number of iterations we should take to perform validation.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon value for learning rate.')

    parser.add_argument('--filename', default=None, type=str, help='Name of results file to be saved.')

    parser.add_argument('--model_name', default=None, type=str, help='Name of model.')
    parser.add_argument('--save_path', default=None, type=str, help='The path where the model is going to be saved.')


    args = parser.parse_args()

    # Number of classes 
    CLASSES = 1
    CLASS_NAMES = [i for i in range(CLASSES)]

    epoch_n = args.epoch_n
    filename = args.filename 

    if filename is None: 
        filename = f'./results/{args.model_name}/{datetime.now()}.txt'

    write_file = open(filename, "w")

    if write_file:
        write_file.write(f'*** args: {args}\n\n')

    # create model 
    collators, model = create_model_and_collator(
        args = args, 
        model_name = args.model_name
    )
    model.to(device)

    # load data 
    data_loaders = create_dataset(
        args = args, collator_fns = collators
    )


    if args.model_name in ['logistic_regression', 'basic_cnn', 'dense_cnn']:
        optim = torch.optim.Adam(params = model.parameters())
    else: 
        optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=args.eps)

    total_steps = len(data_loaders[0]) * args.epoch_n 
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps = total_steps)

    criterion = torch.nn.MSELoss()

    if write_file: 
        write_file.write(f'\nModel:\n {model}\nOptimizer:{optim}\n')

    train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file)

    if write_file:
        write_file.close()



















