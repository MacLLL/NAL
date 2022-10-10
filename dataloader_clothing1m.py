from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch

class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, num_class=14):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.selected_train_labels=[]

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if mode == 'all':
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for i,l in enumerate(lines):
                    img_path = '%s/' % self.root + l[7:]
                    # train_imgs.append(img_path)
                    train_imgs.append((i,img_path))
            # print(train_imgs[:3])
            self.num_raw_example = len(train_imgs)
            # print(self.num_raw_example)
            random.shuffle(train_imgs) #whether shuffle the data
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for id_raw, impath in train_imgs:
                label = self.train_labels[impath]
                # print('label is {}'.format(label))
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append((id_raw,impath))
                    class_num[label] += 1
                    self.selected_train_labels.append(label)
            # print(train_imgs[:10])
            # random.shuffle(self.train_imgs)
        elif mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.test_imgs.append(img_path)
        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == 'all':
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img,target,index
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader():
    def __init__(self, root, batch_size, num_batches, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root

        self.transform_samples = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='all',
                                              num_samples=self.num_batches * self.batch_size )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size ,
                shuffle=True,
                num_workers=self.num_workers)
            return train_loader,np.asarray(train_dataset.selected_train_labels)
        if mode == 'show_samples':
            train_dataset = clothing_dataset(self.root, transform=self.transform_samples, mode='all',
                                              num_samples=self.num_batches * self.batch_size)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return train_loader, np.asarray(train_dataset.selected_train_labels)
        elif mode == 'test':
            test_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'val':
            val_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)
            return val_loader