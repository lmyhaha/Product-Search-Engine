# -*- coding: UTF-8 -*-
# !/usr/bin/env python


import sys, os, lucene, jieba

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
import cv2
from sklearn.externals import joblib
import math
import time
import heapq
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def laptop_branddecide(brand):
    laptop_pinpai = ['华硕', '神舟', 'Apple', '惠普', '戴尔', '宏碁', '机械师', '三星', 'iru', '机械革命', '华为', '联想', 'ITSOHOO',
                     'ThinkPad', '海尔', '小米', '巴克莱', '雷神', '雷蛇',
                     '酷睿', '微软', 'VAIO', 'EVE', 'AVITA', '微星', 'RedmiBook', '苹果', '炫龙', 'LG', '神盾', '掠夺者', 'GPD', '外星人',
                     'YEPO', '麦本本', '荣耀', '技嘉']
    for pinpai in laptop_pinpai:
        if pinpai in brand:
            brand = pinpai
    if '联想' in brand:
        return '联想(Lenovo)'
    elif '华为' in brand:
        return '华为(HUAWEI)'
    elif '三星' in brand:
        return '三星(SAMSUNG)'
    elif 'SONY' in brand:
        return '索尼(SONY)'
    elif '索尼' in brand:
        return '索尼(SONY)'
    return brand


def camera_branddecide(brand):
    if '富士'in brand:
        return '富士(FUJIFILM)'
    elif '尼康' in brand:
        return '尼康(Nikon)'
    elif 'SONY' in brand:
        return '索尼(SONY)'
    elif '松下' in brand:
        return '松下(Panasonic)'
    elif '奥林巴斯' in brand:
        return '奥林巴斯(OLYMPUS)'
    elif '佳能' in brand:
        return '佳能(Canon)'
    elif '索尼' in brand:
        return '索尼(SONY)'
    else:
        return brand


def phone_branddecide(brand):
    if "华为" in brand:
        return '华为(HUAWEI)'
    elif "荣耀" in brand:
        return '荣耀(honor)'
    elif "魅族" in brand:
        return '魅族(MEIZU)'
    elif "三星" in brand:
        return '三星(SAMSUNG)'
    elif "小米" in brand:
        return '小米(mi)'
    elif "Apple" in brand:
        return "苹果(Apple)"
    elif  '联想' in brand:
        return "联想(Lenovo)"
    elif '华硕' in brand:
        return '华硕'
    elif 'SONY' in brand:
        return '索尼(SONY)'
    elif '索尼' in brand:
        return '索尼(SONY)'
    else:
        return brand


def headphone_branddecide(brand):
    if 'beats' in brand:
        return 'Beats'
    elif '华为' in brand:
        return '华为(HUAWEI)'
    elif 'SONY' in brand:
        return '索尼(SONY)'
    elif '小米' in brand:
        return '小米(MI)'
    elif 'OPPO' in brand:
        return 'OPPO'
    elif '苹果' in brand:
        return '苹果(Apple)'
    elif '华硕' in brand:
        return '华硕'
    elif 'HUAWEI' in brand:
        return '华为(HUAWEI)'
    elif '索尼' in brand:
        return '索尼(SONY)'
    return brand


def camera_kind(title):
    if '单反' in title:
        return '单反相机'
    elif '微单' in title:
        return '微单相机'
    elif '运动' in title:
        return '运动相机'
    elif '多功能' in title:
        return '多功能相机'
    elif '数码' in title:
        return '数码相机'
    else:
        return '其他'


def laptop_kind(title):
    if '游戏' in title:
        return '游戏本'
    elif '商' in title:
        return '商务本'
    elif '薄' in title:
        return '轻薄本'
    else:
        return '其他'


def phone_kind(title):
    if '苹果' in title or 'Apple' in title or 'apple' in title:
        return 'IOS手机'
    elif '朵唯' in title or '锤子' in title or '纽曼' in title or '8848' in title or '天语' in title or 'AGM' in title:
        return '其他'
    else:
        return 'Android手机'


def show(li):
    for i in range(len(li)):
        print"title: %s" % li[i]['title']
        print"price: %s" % li[i]['price']
        print"quality: %s" % li[i]['quality']


def relation(li):
    bigli = []
    for path in li:
        dic = {}
        if "suning" in path:

            files = os.listdir(path)
            file = files[0]
            if (file == "1.txt"):
                file = files[1]
            with open(path + "/" + file) as f:
                result = f.readlines()
            dic['url'] = result[0].strip('\n')
            dic['imgurl'] = result[1].strip('\n')
            dic['price'] = float(result[3].strip('\n'))
            star5 = int(result[5].strip('\n'))
            star4 = int(result[6].strip('\n'))
            star3 = int(result[7].strip('\n'))
            star2 = int(result[8].strip('\n'))
            star1 = int(result[9].strip('\n'))
            try:
                quality = (star5 * 5 + star4 * 4 + star3 * 3 + star2 * 2 + star1 * 1) / (
                            star5 + star4 + star3 + star2 + star1 + 0.0)
            except:
                quality = 0
            dic['quality'] = format(20 * quality, '.2f')
            title = result[2].strip('\n')
            dic['title'] = title
            kind = 'none'
            feature1 = ' '
            feature2 = ' '
            feature3 = ' '
            brand = result[10].strip('\n')

            if "camera" in file:
                if '品牌' in brand:
                    brand = brand.split('：')[1]
                else:
                    brand = result[2].strip('\n').split(' ')[0]
                brand=camera_branddecide(brand)
                kind=camera_kind(title)
                for i in range(10, len(result)):
                    row = result[i].strip('\n')
                    if row == "***":
                        break
                    key = row.split('：')[0]

                    if key == '镜头类型':
                        feature1 = row.split('：')[1]
                        if feature1 == '无':
                            feature1 = '无变焦'
                    if key == '颜色':
                        feature2 = row.split('：')[1]
            if "headphone" in file:
                if '品牌' in brand:
                    brand = brand.split('	')[1]
                else:
                    brand = result[2].strip('\n').split(' ')[0]
                brand=headphone_branddecide(brand)
                for i in range(10, len(result)):
                    row = result[i].strip('\n')
                    if row == "***":
                        break
                    key = row.split('	')[0]
                    if key == '佩戴方式':
                        kind = row.split('	')[1]
                    if key == '耳机频响范围':
                        feature1 = row.split('	')[1]
                    if key == '颜色':
                        feature2 = row.split('	')[1]
                    if key == '产地':
                        feature3 = row.split('	')[1]
                kind=kind+'耳机'
            if "phone" in file and "headphone" not in file:
                if '品牌' in brand:
                    brand = brand.split('	')[1]
                else:
                    brand = result[2].strip('\n').split(' ')[0]
                brand=phone_branddecide(brand)
                kind=phone_kind(title)
                for i in range(10, len(result)):
                    row = result[i].strip('\n')
                    if row == "***":
                        break
                    key = row.split('	')[0]

                    if key == 'CPU型号':
                        feature1 = row.split('	')[1]
                    if key == '手机操作系统':
                        feature2 = row.split('	')[1]
                    if key == '运行内存':
                        feature3 = row.split('	')[1]+'运行内存'

            if "laptop" in file:
                if '品牌' in brand:
                    brand = brand.split('	')[1]
                else:
                    brand = result[2].strip('\n').split(' ')[0]
                brand = laptop_branddecide(brand)
                kind = laptop_kind(title)
                for i in range(10, len(result)):
                    row = result[i].strip('\n')
                    if row == "***":
                        break
                    key = row.split('	')[0]

                    if key == '产品定位':
                        values = row.split('	')[1]
                        values2 = values.split(',')
                        if (len(values2) == 1):
                            feature1 = values2[0]
                        elif (len(values2) == 2):
                            feature1 = values2[0]
                            feature2 = values2[1]
                        else:
                            feature1 = values2[0]
                            feature2 = values2[1]
                            feature3 = values2[2]
            dic['kind'] = kind
            dic['brand'] = brand
            dic['features'] = [feature1, feature2, feature3]
            bigli.append(dic)

        else:
            with open(path) as f:
                result = f.readlines()
            url = result[0].strip('\n').decode('gbk', 'ignore')
            imgurl = result[1].strip('\n').decode('gbk', 'ignore')
            title = result[2].strip('\n').decode('gbk', 'ignore')
            price = result[3].strip('\n').decode('gbk', 'ignore')
            star5 = int(result[6].strip('\n').decode('gbk', 'ignore'))
            star3 = int(result[7].strip('\n').decode('gbk', 'ignore'))
            star1 = int(result[8].strip('\n').decode('gbk', 'ignore'))
            try:
                quality = (5 * star5 + 3 * star3 + star1) / (star5 + star3 + star1 + 0.0)
            except:
                quality = 0
            dic['url'] = url
            dic['imgurl'] = imgurl
            dic['price'] = float(price)
            dic['quality'] = format(20 * quality, '.2f')
            title = title.encode('utf-8', 'ignore')
            dic['title'] = title
            kind = 'none'
            feature1 = ' '
            feature2 = ' '
            feature3 = ' '
            brand = title.split(' ')[0]
            if 'camera' in path:
                kind = camera_kind(title)
                for i in range(9, len(result)):
                    row = result[i].strip('\n').decode('gbk', 'ignore').encode('utf-8', 'ignore')
                    if row == "***":
                        break
                    key = row.split('	')[0]
                    if key == "品牌":
                        brand = row.split('	')[1]


                    if key == "场景模式":
                        value = row.split('	')[1]
                        r = value.split('；')
                        if len(r) == 1:
                            feature1 = r[0]
                        elif len(r) == 2:
                            feature1 = r[0]
                            feature2 = r[1]
                        else:
                            feature1 = r[0]
                            feature2 = r[1]
                            feature3 = r[2]
                brand = camera_branddecide(brand)
            if 'headphone' in path:
                for i in range(9, len(result)):
                    row = result[i].strip('\n').decode('gbk', 'ignore').encode('utf-8', 'ignore')
                    if row == "***":
                        break
                    key = row.split('	')[0]
                    if key == "品牌":
                        brand = row.split('	')[1]

                    if key == "线控功能":
                        kind = row.split('	')[1]
                    if key == "音频接口":
                        feature1 = row.split('	')[1]
                        if feature1 == '无':
                            feature1 = '无音频接口'
                    if key == "接口类型":
                        feature2 = row.split('	')[1]
                    if key == "颜色":
                        feature3 = row.split('	')[1]
                brand = headphone_branddecide(brand)
                kind=kind+'耳机'
            if "phone" in path and 'headphone' not in path:
                kind = phone_kind(title)
                for i in range(9, len(result)):
                    row = result[i].strip('\n').decode('gbk', 'ignore').encode('utf-8', 'ignore')
                    if row == "***":
                        break
                    key = row.split('	')[0]
                    if key == "品牌":
                        brand = row.split('	')[1]


                    if key == "双卡机类型":
                        feature1 = row.split('	')[1]
                    if key == "屏幕材质类型":
                        feature2 = row.split('	')[1]
                    if key == "运行内存":
                        feature3 = row.split('	')[1]+'运行内存'
                brand = phone_branddecide(brand)
            if "laptop" in path:
                brand = laptop_branddecide(brand)
                kind = laptop_kind(title)
                for i in range(9, len(result)):
                    row = result[i].strip('\n').decode('gbk', 'ignore').encode('utf-8', 'ignore')
                    if row == "***":
                        break
                    key = row.split('	')[0]

                    if key == "屏幕类型":
                        feature1 = row.split('	')[1]
                    if key == "局域网":
                        feature2 = row.split('	')[1]
                    if key == "电池":
                        feature3 = row.split('	')[1]
            dic['kind'] = kind
            dic['brand'] = brand
            dic['features'] = [feature1, feature2, feature3]
            bigli.append(dic)
    return bigli


def priceup(li):
    return heapq.nsmallest(len(li), li, key=lambda x: x["price"])


def pricedown(li):
    return heapq.nlargest(len(li), li, key=lambda x: x["price"])


def qualitydown(li):
    return heapq.nlargest(len(li), li, key=lambda x: x["quality"])


def sort(li):
    li1 = relation(li)
    li2 = priceup(li1)
    li3 = pricedown(li1)
    li4 = qualitydown(li1)
    return [li1, li2, li3, li4]


def countkind(li):
    dic = {}
    for i in range(len(li)):
        kind = li[i]['kind']
        if kind != 'none耳机':
            if kind not in dic.keys():
                dic[kind] = 1
            else:
                dic[kind] += 1
    li = sorted(dic, key=dic.__getitem__, reverse=True)
    kind1 = kind2 = kind3 = ''
    if (len(li) > 2):
        kind1 = li[0]
        kind2 = li[1]
        kind3 = li[2]
    elif (len(li) == 2):
        kind1 = li[0]
        kind2 = li[1]
    else:
        kind1 = li[0]
    return [kind1, kind2, kind3]


def countbrand(li):
    dic = {}
    for i in range(len(li)):
        kind = li[i]['brand']
        if kind != 'none' and kind!='其他':
            if kind not in dic.keys():
                dic[kind] = 1
            else:
                dic[kind] += 1
    li = sorted(dic, key=dic.__getitem__, reverse=True)
    kind1 = kind2 = kind3 = ''
    if (len(li) > 2):
        kind1 = li[0]
        kind2 = li[1]
        kind3 = li[2]
    elif (len(li) == 2):
        kind1 = li[0]
        kind2 = li[1]
    else:
        kind1 = li[0]
    return [kind1, kind2, kind3]


def countfeatures(li):
    dic = {}
    for i in range(len(li)):
        features = li[i]['features']
        for feature in features:
            if feature != ' ' and feature!='其它' and '为准' not in feature and '其他' not in feature:
                if feature not in dic.keys():
                    dic[feature] = 1
                else:
                    dic[feature] += 1
    f0 = dic.keys()[0]
    for data in dic.keys():
        if dic[data] < dic[f0]:
            f0 = data
    feature1 = f0
    feature2 = f0
    feature3 = f0
    for f1 in dic.keys():
        if dic[f1] > dic[feature1]:
            feature1 = f1
    for f2 in dic.keys():
        if dic[f2] > dic[feature2] and f2 != feature1:
            feature2 = f2
    for f3 in dic.keys():
        if dic[f3] > dic[feature3] and f3 != feature1 and f3 != feature2:
            feature3 = f3
    return [feature1, feature2, feature3]


def count(li):
    li1 = countkind(li)
    li2 = countbrand(li)
    li3 = countfeatures(li)
    return [li1, li2, li3]


def find(filename, path):
    if ('-' in filename):
        f0 = filename.split('-')[0]
        f1 = f0.split('_')[1]
        path1 = path + "/suning/" + f1
        files = os.listdir(path1)
        for file in files:
            li = os.listdir(path1 + "/" + file)
            if filename in li:
                return path1 + "/" + file
    else:
        f0 = filename.split('_')[1]
        f1 = f0.split('.')[0]
        path1 = path + "/jingdong/" + f1 + "/" + filename
        return path1


def func(command):
    STORE_DIR = "index-n"
    try:
        vm_env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    except:
        vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    directory = SimpleFSDirectory(File(STORE_DIR))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
    command = ' '.join(jieba.cut(command))
    if command == '':
        return
    query = QueryParser(Version.LUCENE_CURRENT, "head",
                        analyzer).parse(command)
    scoreDocs = searcher.search(query, 120).scoreDocs
    print "%s total matching documents." % len(scoreDocs)
    li = []
    path = "data"
    for i, scoreDoc in enumerate(scoreDocs):
        doc = searcher.doc(scoreDoc.doc)
        li.append(path + doc.get("lujing"))
    bigli = relation(li)
    li1 = sort(li)
    li2 = count(bigli)
    return li1, li2


def default_loader(path):
    in_img = Image.open(path).convert('RGB')
    s = 60
    out_img = in_img.resize((s, s), Image.ANTIALIAS)
    return out_img


class MyDataset(Dataset):
    def __init__(self, jpg, transform=None, target_transform=None, loader=default_loader):
        imgs = [(jpg, int(-1))]
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# -----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


def judge(root):
    test_data = MyDataset(jpg=root, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    model = Net()
    restore_model_path = './checkpoint/py2_ckpt_20.pth'
    model.load_state_dict(torch.load(restore_model_path)['net'])
    # evaluation--------------------------------
    model.eval()
    for batch_x, batch_y in test_loader:
        with torch.no_grad():
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        pred = torch.max(out, 1)[1]
        return int(pred)


def distance(x, y):
    dis = 0
    for i in range(len(x)):
        dis += math.pow(x[i] - y[i], 2)
    return dis


def get_match(root, number, path):
    pretend = judge(root)
    Image = cv2.imread(root)
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.ORB_create(80)
    kps, descs = descriptor.detectAndCompute(gray, None)
    descs = descs.astype("int")
    store_set = ["model/py2_camera.model", "model/py2_headphone.model",
                 "model/py2_laptop.model", "model/py2_phone.model"]
    store_set = store_set[pretend]
    res = joblib.load(store_set)
    predict = res.predict(descs)
    vector = []
    for i in range(16):
        vector.append(0)
    for one in predict:
        vector[int(one)] += 1

    clumsy = []
    dataset = "img_kmeans"
    for root, dir_names, files in os.walk(dataset):
        target = "camera.txt"
        if pretend == 1:
            target = "headphone.txt"
        elif pretend == 2:
            target = "laptop.txt"
        elif pretend == 3:
            target = "phone.txt"
        root2 = root + "/" + target
        with open(root2, "r") as f:
            for line in f:
                feature_vector = []
                line = line.split(" ")[:-1]
                feature_dir = line[0]
                for i in range(1, len(line)):
                    feature_vector.append(int(line[i]))
                clumsy.append({"distance": distance(vector, feature_vector), "id": feature_dir})
    cheap = heapq.nsmallest(number, clumsy, key=lambda s: s["distance"])
    result = []
    for one in cheap:
        result.append(one["id"])
    li = []
    for i in range(len(result)):
        li.append(find(result[i], path))
    return li


def func2(get_post):
    path = 'data'
    num = 120
    li = get_match(get_post, num, path)
    li=list(set(li))
    li1 = sort(li)
    bigli = relation(li)
    li2 = count(bigli)
    return li1, li2


if __name__ == '__main__':
    get_post = "yes.jpg"
    func2(get_post)
