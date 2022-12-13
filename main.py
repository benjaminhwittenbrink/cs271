# standard libraries 
import os 
import argparse 
import random 
import pandas as pd
import numpy as np 
import collections 
from tqdm import tqdm 
from datetime import datetime
import pickle 

# PyTorch libraries 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
# from torchvision.datasets import DatasetFolder
from torchvision import models, transforms 

# Hugging Face datasets 
#import datasets 

# Transformers libraries 
# from transformers import TrainingArguments, Trainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import get_linear_schedule_with_warmup 

# import sklearn models 
# from sklearn.neighbors import KNeighborsClassifier

# simple models
from models import LogisticRegression, BasicCNNModel, DenseCNNModel, BasicCNNCountryModel, ViTCountryModel, ViTMosaiksModel
from SatelliteImageDataset import SatelliteImageDataset, SatelliteImageMetadataDataset, SatelliteImageMosaiksDataset

from sklearn.metrics import confusion_matrix


RANDOM_SEED = 231 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class ImageClassificationCollator:
    def __init__(self, feature_extractor, transforms = False, metadata = False, mosaiks = False): 
        self.feature_extractor = feature_extractor
        self.transforms = transforms 
        self.metadata = metadata
        self.mosaiks = mosaiks

    def __call__(self, batch):
        if self.transforms: 
            transformed = [self.feature_extractor(x[0].cpu().detach().numpy()) for x in batch]
            encodings = {"pixel_values":torch.stack(transformed)}
        else: 
            encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')   
        encodings['labels'] = torch.tensor([x[1] for x in batch],  dtype=torch.long)
        
        if self.metadata: 
            if "country" in self.metadata:
                encodings['country'] = torch.tensor([x[2] for x in batch])
        elif self.mosaiks: 
            encodings['mosaiks_features'] = torch.tensor(np.array([x[2] for x in batch]), dtype = torch.float32)

        return encodings

# create model and collator
def create_model_and_collator(args, model_name, metadata = None, cnt_id_map = None):

    if metadata:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        collator = ImageClassificationCollator(feature_extractor, metadata=metadata)
        collators = (collator, collator)
        if model_name in ['basic_cnn']:
            model = BasicCNNCountryModel(n_classes=CLASSES, cnt_id_map = cnt_id_map, num_country_embeddings=len(cnt_id_map))
        elif model_name == "ViT":
            model = ViTCountryModel(n_classes=CLASSES, cnt_id_map = cnt_id_map, num_country_embeddings=len(cnt_id_map))
    elif args.mosaiks: 
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        collator = ImageClassificationCollator(feature_extractor, mosaiks=args.mosaiks)
        collators = (collator, collator)
        model = ViTMosaiksModel(n_classes=CLASSES, mosaiks_dim = 3999)
    elif model_name == "ViT":
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

    elif model_name in ['basic_cnn', 'basic_cnn_novit', 'dense_cnn', 'logistic_regression']:
        # ADD IN transforms though feature extractor might be easier 
        if "novit" in model_name:
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
        else: 
            feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            collator = ImageClassificationCollator(feature_extractor)
            collators = (collator, collator)
        # TODO: add support for model!
        if model_name == "logistic_regression":
            model = LogisticRegression(n_classes=CLASSES)
        elif "basic_cnn" in model_name:
            model = BasicCNNModel(n_classes=CLASSES)
        elif model_name == "dense_cnn":
            model = DenseCNNModel(n_classes=CLASSES)

    else: 
        raise NotImplementedError

    print(f'Model name: {model_name}')

    return collators, model 


def create_dataset(args, collator_fns, metadata = None, cnt_id_map = None, val_split = 0.15):

    def npy_loader(path):
        sample = torch.from_numpy(np.load(path))
        return sample 
    
    # load in dataset frmom directory 
    if metadata:
        dataset = SatelliteImageMetadataDataset(
            root = args.data_dir, 
            csv_path = args.csv_file, 
            outcome = args.outcome, 
            loader = npy_loader
        )
    elif args.mosaiks: 
        dataset = SatelliteImageMosaiksDataset(
            root = args.data_dir, 
            csv_path = args.csv_file, 
            outcome = args.outcome, 
            mosaiks_csv_path = args.mosaiks_csv_file, 
            loader = npy_loader
        )
    else:
        dataset = SatelliteImageDataset(
            root = args.data_dir, 
            csv_path = args.csv_file, 
            outcome = args.outcome, 
            loader = npy_loader
        )

    # IDEALLY we would like same sampling...

    # split up into train val data 
    if os.path.isfile("indices_perm.npy"):
        indices = np.load("indices_perm.npy") 
    else:
        indices = torch.randperm(len(dataset)).tolist()
        np.save("indices_perm.npy", indices)
    
    n_val = int(np.floor(len(indices) * val_split))
    train_ds = torch.utils.data.Subset(dataset, indices[:-n_val])
    val_ds = torch.utils.data.Subset(dataset, indices[-n_val:])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator_fns[0], shuffle = 1)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator_fns[1], shuffle=0)

    return [train_dl, val_dl]

def dataset_statistics(args, dataset_loader):
    label_stats = collections.Counter()
    for i, batch in enumerate(dataset_loader):
        inputs, labels = batch['pixel_values'], batch['labels']
        labels = labels.cpu().numpy().flatten()
        label_stats += collections.Counter(labels)
    return label_stats


def measure_accuracy(outputs, labels):
    preds = np.argmax(outputs, axis = 1).flatten()
    labels = labels.flatten()
    correct = np.sum(preds == labels)
    c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
    return correct, len(labels), c_matrix 

def validation(args, val_loader, model, criterion, metadata, device, name = 'Validation', write_file=None):

    model.eval()
    total_loss = 0. 
    total_correct = 0 
    total_sample = 0 
    total_confusion = np.zeros((CLASSES, CLASSES))

    for i, batch in enumerate(tqdm(val_loader)):
        inputs, labels = batch['pixel_values'], batch['labels'] 
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if metadata: 
                country = batch['country'].to(device)
                outputs = model(inputs, country)
            elif args.mosaiks: 
                mosaiks_features = batch['mosaiks_features'].to(device)
                outputs = model(inputs, mosaiks_features)
            elif args.model_name in [
            'basic_cnn', 'basic_cnn_novit', 'dense_cnn', 'logistic_regression',
            'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
            ]:
                outputs = model(inputs)
            else: 
                outputs = model(inputs)['logits'] 

        loss = criterion(outputs, labels)

        logits = outputs 
        total_loss += loss.cpu().item()

        correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), labels.cpu().numpy())
        total_correct += correct_n 
        total_sample += sample_n 
        total_confusion += c_matrix 

    bal_acc = np.mean( np.diag(total_confusion) / np.sum(total_confusion, 1) )
    print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
    print(f'*** Weighted accuracy on the {name} set: {bal_acc}')
    print(f'*** Confusion matrix:\n{total_confusion}')
    if write_file:
        write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
        write_file.write(f'*** Weighted accuracy on the {name} set: {np.mean( np.diag(total_confusion) / np.sum(total_confusion, 1) )}')
        write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')

    return total_loss, float(total_correct / total_sample) * 100, bal_acc



def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, metadata, device, write_file=None):
    print("\n>>> Training starts...")

    if write_file: 
        write_file.write("\n>>> Training starts...")

    model.train()

    best_val_acc = 0
    for epoch in range(epoch_n):
        print("*** Epoch:", epoch)
        total_train_loss = 0. 
        total_correct = 0
        total_sample = 0

        for i, batch in enumerate(tqdm(data_loaders[0])): 
            optim.zero_grad()
            inputs, labels = batch['pixel_values'], batch['labels'] 
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward pass 
            if metadata: 
                country = batch['country'].to(device)
                outputs = model(inputs, country)
            elif args.mosaiks: 
                mosaiks_features = batch['mosaiks_features'].to(device)
                outputs = model(inputs, mosaiks_features)
            elif args.model_name in [
                'basic_cnn', 'basic_cnn_novit', 'dense_cnn', 'logistic_regression', 
                'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
            ]:
                outputs = model(inputs)
            else: 
                outputs = model(inputs)['logits']

            loss = criterion(outputs, labels)
            logits = outputs
            correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().detach().numpy(), labels.cpu().detach().numpy())
            total_correct += correct_n 
            total_sample += sample_n 

            total_train_loss += loss.cpu().item()

            # backward pass 
            loss.backward()
            optim.step()

            if scheduler: scheduler.step()

            if i % args.val_every == 0: 
                print(f'*** Average Loss: {total_train_loss / (i+1)}')
                print(f'*** Running accuracy on the train set: {total_correct/total_sample}')
                if write_file:
                    write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')
                    write_file.write(f'*** Loss: {loss}\n')
                    write_file.write(f'*** Running accuracy on the train set: {total_correct/total_sample}\n')

                _, val_acc_unbal, val_acc = validation(args, data_loaders[1], model, criterion, metadata, device, write_file=write_file)

                model.train()

                if best_val_acc < val_acc: 
                    best_val_acc = val_acc 

                    if args.save_path:
                        if args.model_name in ['ViT']:
                            if metadata: 
                                with open(args.save_path + "/" + "ViT_metadata.pkl", "wb") as f: 
                                    pickle.dump(model, f)
                            elif args.mosaiks:
                                with open(args.save_path + "/" + "ViT_mosaiks.pkl", "wb") as f: 
                                    pickle.dump(model, f)                                 
                            else: 
                                model.save_pretrained(args.save_path)
                        else: 
                            torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':

    # set device to GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    # example args would be 
    # main.py --data_dir west_africa_npy --csv_file west_africa_df --outcome Mean_BMI_bin  --n_classes 3
    parser.add_argument('--data_dir', type=str, help='Image data location.')
    parser.add_argument('--csv_file', type=str, help='CSV file with labels.')
    parser.add_argument('--outcome', type=str, help='Label of outcome variable in df.')
    parser.add_argument('--n_classes', type=int, help='Number of classes in outcome variable.')	
    parser.add_argument('--mosaiks_csv_file', type=str, help='CSV file for mosaiks features.')

    parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
    parser.add_argument('--epoch_n', default=10, type=int, help='Number of epochs for training.')
    parser.add_argument('--val_every', default=200, type=int, help="Number of iterations we should take to perform validation.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon value for learning rate.')

    parser.add_argument('--filename', default=None, type=str, help='Name of results file to be saved.')

    parser.add_argument('--model_name', default=None, type=str, help='Name of model.')
    parser.add_argument('--mosaiks', action='store_true', help="Whether to include mosaiks features.")
    parser.add_argument('--metadata', action='store_true', help="Whether to include metadata.")
    parser.add_argument('--save_path', default=None, type=str, help='The path where the model is going to be saved.')

    # parser.add_argument('--n_filters', type=int, default=25, help='Number of filters in the CNN (if applicable)')
    # parser.add_argument('--filter_sizes', type=int, nargs='+', action='append', default=[[3,4,5], [5,6,7], [7,9,11]], help='Filter sizes for the CNN (if applicable).')


    args = parser.parse_args()

    # Number of classes 
    CLASSES = args.n_classes
    CLASS_NAMES = [i for i in range(CLASSES)]
    
    epoch_n = args.epoch_n
    filename = args.filename 

    # read df 
    df = pd.read_csv(args.csv_file + ".csv")

    if args.metadata and args.mosaiks: 
        raise NotImplementedError("Functionality for both mosaiks and metadata has not been implemented yet.")

    cnt_id_map = None
    metadata = None
    if args.metadata: 
        metadata = ["country"]
        unique_countries = list(set(df["country"]))
        unique_countries_int = [int(str(ord(c[0])) + str(ord(c[1]))) for c in unique_countries]
        cnt_id_map = {float(v):k for k, v in enumerate(set(unique_countries_int))}

    if filename is None: 
        filename = f'./results/{args.model_name}/{datetime.now()}.txt'

    write_file = open(filename, "w")

    if write_file:
        write_file.write(f'*** args: {args}\n\n')

    # create model 
    collators, model = create_model_and_collator(
        args = args, 
        model_name = args.model_name, 
        metadata=metadata, cnt_id_map = cnt_id_map

    )
    model.to(device)

    # load data 
    data_loaders = create_dataset(
        args = args, collator_fns = collators, metadata = metadata, cnt_id_map = cnt_id_map
    )

    # train_label_stats = dataset_statistics(args, data_loaders[0])
    # val_label_stats = dataset_statistics(args, data_loaders[1])
    # print(f'*** Training set label statistics: {train_label_stats}')
    # print(f'*** Validation set label statistics: {val_label_stats}')

    # if write_file:
    #     write_file.write(f'*** Training set label statistics: {train_label_stats}')
    #     write_file.write(f'*** Validation set label statistics: {val_label_stats}')	


    if args.model_name in ['logistic_regression', 'basic_cnn', 'dense_cnn']:
        optim = torch.optim.Adam(params = model.parameters())
    else: 
        optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=args.eps)

    total_steps = len(data_loaders[0]) * args.epoch_n 
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps = total_steps)

    # get class weights 
    class_weights = 1 - df[args.outcome].value_counts(normalize=True).sort_index()
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if write_file: 
        write_file.write(f'\nModel:\n {model}\nOptimizer:{optim}\n')

    train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, metadata, device, write_file)

    if write_file:
        write_file.close()



















