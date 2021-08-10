from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import os
import cv2
from torch.autograd import Variable
import shutil
import telebot


class SiamesNetwork(nn.Module):
    def __init__(self):
        super(SiamesNetwork, self).__init__()
        self.cnn = EfficientNet.from_name("efficientnet-b0")
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


transform = albu.Compose([
            albu.Resize(256, 256),
            albu.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ToTensorV2()
])
classes = ['bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='data/best.pt')
net = SiamesNetwork()
net.load_state_dict(torch.load("Efficient_siames.pth", map_location=device))
net.eval()
net.to(device)


def load_img(path2img):
    image = cv2.imread(path2img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_compatibility(img_path):
    image = cv2.imread(img_path)
    result = model([img_path], size=1280)
    res = result.xyxy[0].tolist()
    img_paths, image_classes = [], []
    os.makedirs('/content/data_tmp', exist_ok=True)
    shutil.rmtree("/content/data_tmp")
    os.makedirs('/content/data_tmp/0')
    ignore_classes = [1, 3, 6]
    for i in range(len(res)):
        if res[i][4] > 0.5 and int(res[i][5]) not in ignore_classes:
            res[i] = [int(x) for x in res[i]]
            #cut_img = image[max(res[i][1]-10, 0):min(res[i][3]+10, image.shape[0]), max(res[i][0], 0):min(res[i][2]+10, image.shape[1])]
            cut_img = image[res[i][1]:res[i][3], res[i][0]:res[i][2]]
            if not os.path.exists(f'/content/data_tmp/0/{str(int(res[i][5]))}.jpg'):
                cv2.imwrite(f'/content/data_tmp/0/{str(int(res[i][5]))}.jpg', cut_img)
                img_paths.append(f'/content/data_tmp/0/{str(int(res[i][5]))}.jpg')
                image_classes.append(classes[int(res[i][5])])
    all = 0
    num = 0
    transform = albu.Compose([
                albu.Resize(256, 256),
                albu.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ToTensorV2()
    ])
    answer = ''
    for i in range(len(img_paths)):
        image1 = load_img(img_paths[i])
        image1 = transform(image=image1)['image']
        for j in range(i+1, len(img_paths)):
            image2 = load_img(img_paths[j])
            image2 = transform(image=image2)['image']
            image1, image2 = Variable(image1.to(device)), Variable(image2.to(device))
            pred = net(image1.unsqueeze(0), image2.unsqueeze(0))
            all += pred
            num += 1
            answer += f'{image_classes[i]} {image_classes[j]} {int(pred*100)}%\n'
            print(pred)
            print(answer)
    if num == 0:
        return 0
    else:
        answer += f'summary: {int(all/num*100)}%'
        return answer


bot = telebot.TeleBot("") # Your bot token


@bot.message_handler(content_types=['photo'])
def get_pred(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = '/content/data/' + file_info.file_path.split('/')[-1]
    with open(src, "wb") as new_file:
        new_file.write(downloaded_file)

    outputs = get_compatibility(src)
    bot.reply_to(message, outputs)

bot.polling(none_stop=True, interval=1)
