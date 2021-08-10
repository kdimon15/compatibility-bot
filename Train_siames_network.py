import math
import os
import random
import time
import albumentations as albu
import cv2
import torch
import torch.nn as nn
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from contextlib import contextmanager

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("data/detected_labels.csv")
paths = df.path.values.tolist()
lbs = df.label.values.tolist()
dic = {}
for i in range(len(paths)):
    dic[paths[i]] = lbs[i]


class CFG:
    size = 256
    num_epochs = 100
    batch_size = 52
    lr = 1e-4


image1, image2, labels = [], [], []
classes1, classes2 = [], []
main_root = os.listdir("data")
ignore_classes = [1, 3, 6]


for root in tqdm(main_root):
    was = []
    for i in range(len(os.listdir(f"data/{root}"))):
        pap = os.listdir(f'data/{root}')
        cls1 = dic[f'data/{root}/{str(i)}.jpg']
        if cls1 not in was:
            was.append(cls1)
            for j in range(i+1, len(pap)):
                cls2 = dic[f'data/{root}/{str(j)}.jpg']
                if cls1 != cls2 and cls1 not in ignore_classes and cls2 not in ignore_classes:
                    image1.append(f"{root}/{str(i)}.jpg")
                    image2.append(f"{root}/{str(j)}.jpg")
                    classes1.append(cls1)
                    classes2.append(cls2)
                    labels.append(1)
                    for k in range(10):
                        r = random.choice(main_root)
                        if len(os.listdir(f'dat/{r}')) > 0:
                            rand = random.randint(0, len(os.listdir(f'data/{r}'))-1)
                            path1 = random.choice(pap)
                            cls4 = dic[f'data/{root}/{path1}']
                            cls3 = dic[f'data/{r}/{str(rand)}.jpg']
                            if len(os.listdir(f'data/{r}')) > 0 and cls3 not in ignore_classes and cls4 not in ignore_classes and cls3 != cls4:
                                image1.append(f'{root}/{path1}')
                                image2.append(f'{r}/{str(rand)}.jpg')
                                labels.append(0)
                                classes1.append(cls4)
                                classes2.append(cls3)

train_df = pd.DataFrame({
    "image1": image1,
    "image2": image2,
    "label": labels,
    "class1": classes1,
    "class2": classes2
})


class SiamesNetwork(nn.Module):
    def __init__(self):
        super(SiamesNetwork, self).__init__()
        self.cnn = EfficientNet.from_pretrained("efficientnet-b0")
        num_features = self.cnn._fc.in_features
        self.cnn._fc = nn.Linear(num_features, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dis = torch.abs(output1 - output2)
        out = self.fc(dis)
        return out


TRAIN_DIR = "data"
OUTPUT_DIR = "./"


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image1 = df.image1.values
        self.image2 = df.image2.values
        self.labels = df.label.values
        self.transform = transform

    def __getitem__(self, idx):
        image1 = cv2.imread(f'{TRAIN_DIR}/{self.image1[idx]}')
        image2 = cv2.imread(f'{TRAIN_DIR}/{self.image2[idx]}')
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        if self.transform:
            image1 = self.transform(image=image1)['image']
            image2 = self.transform(image=image2)['image']
        label = torch.tensor(self.labels[idx]).float()
        return image1, image2, label

    def __len__(self):
        return len(self.df)


def get_transforms(data=''):
    if data == 'train':
        return albu.Compose([
                             albu.Resize(CFG.size, CFG.size),
                             albu.HorizontalFlip(),
                             albu.CLAHE(p=0.3),
                             albu.Blur(blur_limit=7, p=0.4),
                             albu.RandomBrightnessContrast(p=0.5),
                             albu.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                             ToTensorV2()
        ])
    elif data == 'valid':
        return albu.Compose([
                             albu.Resize(CFG.size, CFG.size),
                             albu.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                             ToTensorV2()
        ])


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file=OUTPUT_DIR + "train.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=42)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    model.to(device)
    start = end = time.time()
    global_step = 0
    for step, (images1, images2, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        images1, images2, labels = Variable(images1.to(device)), Variable(images2.to(device)), Variable(
            labels.to(device))

        # images2 = images1.to(device)
        # images2 = images2.to(device)
        # labels = labels.view(-1, 1)
        batch_size = labels.size(0)

        output = model(images1, images2)
        output = output.reshape(CFG.batch_size)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # record loss
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg


train_dataset = TrainDataset(train_df, get_transforms(data='train'))

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

net = SiamesNetwork()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=CFG.lr, amsgrad=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
criterion = nn.MSELoss()

for epoch in range(CFG.num_epochs):
    print(f"Epoch {epoch + 1}")
    start_time = time.time()

    avg_loss = train_fn(train_loader, net, criterion, optimizer, epoch, device)
    scheduler.step(avg_loss)

    elapsed = time.time() - start_time

    LOGGER.info(f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s")

    torch.save(net.state_dict(), f"Siames_network.pth")

