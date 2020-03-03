import torch
import os, glob
import random, csv
import visdom
import time

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms as transforms

root = '../pytorch_learning_dataset/pokeman'


class DIY_dataset(Dataset):
    def __init__(self, root, resize, mode):
        super(DIY_dataset, self).__init__()

        self.root = root
        self.resize = resize
        self.name2label = {}  # 标签编码
        for name in sorted(os.listdir(os.path.join(root))):  # listdir 返回顺序不定
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)
        # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'training':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'validation':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        # 创建或者读取csv
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 这里根据路径获取类别
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.JPG'))

            print(len(images), images[0])
            # 1167 ../pytorch_learning_dataset/pokeman/bulbasaur/00000158.png

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]  # ../pytorch_learning_dataset/pokeman/bulbasaur/00000158.png
                    label = self.name2label[name]
                    # e.g. ../pytorch_learning_dataset/pokeman/charmander/00000194.jpg,1
                    writer.writerow([img, label])
                print('writen into csv file: ', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(2)

        x_denormalized = x * std + mean
        return x_denormalized

    def __getitem__(self, item):
        # item [0, len(images)]
        img, label = self.images[item], self.labels[item]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),  # string path => image
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)

        return img, label


def main():

    viz =visdom.Visdom()

    # method 1 仅适用于图片按照类别规整保存的情况  可以使用函数
    # db = torchvision.datasets.ImageFolder(root=root, transform=tf)
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    db = torchvision.datasets.ImageFolder(root=root, transform=tf)
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    print(db.class_to_idx)

    for x,y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch-img'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)

    # method 2
    # db = torchvision.datasets.ImageFolder(root=root, transforms=tf)
    # db = DIY_dataset(root, 224, 'training')
    # # x, y = next(iter(db))
    # # print('sample: ', x.shape, y.shape, y)
    # # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # for x,y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch-img'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)

if __name__ == '__main__':
    main()
