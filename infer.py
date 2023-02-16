# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2023-02-10 15:07:46
# @Last Modified by:   fyr91
# @Last Modified time: 2023-02-13 14:57:10
import clip
import torch
import torch.nn as nn
from datasets import ActionFrames
from torch.utils.data import DataLoader
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_transform
from utils.Text_Prompt import text_prompt
import time


num_seg = 1
sim_header = "Transf" #Transf   meanP  LSTM  Conv_1D  Transf_cls
pretrain = "weights/vit-b-16-16f.pt"
batch_size = 8
data_workers = 8
input_size = 224
label_data = [
    "falling", # fall
    "fighting", # fight
    "exposed breasts", # nudity
    "exposed anus", # nudity
    "exposed genitalia", # nudity
    "topless", # nudity
    "fighting a doctor", # fight doctor
    "fighting a nurse", # fight doctor
    "suicide by hanging", # suicide
    "climbing over window", # climb over
    "climbing over wall", # climb over
    "gathering", # gathering
    "destroy public facilities", # violance
    "violance behaviour" # violance
]

video_segs = [[
        "test_data/001.jpg",
        "test_data/002.jpg",
        "test_data/003.jpg",
        "test_data/004.jpg",
        "test_data/005.jpg",
        "test_data/006.jpg",
        # "test_data/hurtdoc/img_017.jpg",
        # "test_data/hurtdoc/img_018.jpg",
        # "test_data/hurtdoc/img_019.jpg",
        # "test_data/hurtdoc/img_020.jpg",
        # "test_data/hurtdoc/img_021.jpg",
        # "test_data/hurtdoc/img_022.jpg",
        # "test_data/hurtdoc/img_023.jpg",
        # "test_data/hurtdoc/img_024.jpg",
        # "test_data/hurtdoc/img_025.jpg",
    ]]


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_state_dict = clip.load("weights/ViT-B-16.pt", device=device, jit=False, T=num_seg, tsm=False, dropout=0.0, emb_dropout=0.0)

    fusion_model = visual_prompt(sim_header, clip_state_dict, num_seg)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    # half precision
    clip.model.convert_weights(model_text)  
    clip.model.convert_weights(model_image)

    print("loading checkpoint")
    checkpoint = torch.load(pretrain)
    model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])

    # evaluaiton mode
    model.eval()
    fusion_model.eval()

    print("loading action labels")
    # data = Action_DATASETS(file_list, transform_data)
    classes, num_text_aug, text_dict = text_prompt(label_data)
    # print(text_dict)

    print("generating label features")
    with torch.no_grad():
        t0 = time.time()
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        # print(text_features.size())
        t1 = time.time()
        print(f"text features generated in {t1-t0}")
    
    print("prepare dataLoader")
    data_transform = get_transform(input_size)
    data = ActionFrames(video_segs, data_transform)
    loader = DataLoader(data, batch_size=batch_size, num_workers=data_workers, 
        shuffle=False, pin_memory=True, drop_last=False)
    
    print("predicting frames")
    with torch.no_grad():
        for ii, image in enumerate(loader):
            t2 = time.time()
            image = image.view((-1, num_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            # print(values_5, indices_5)
            t3 = time.time()
            for i in range(b):
                print(f"frame {i}:") 
                for j in range(5):
                    if values_5[i][j] > 0.3:
                        print(f"{label_data[indices_5[i][j]]} - {values_5[i][j]}")
            print(f"video seg features generated in {t3-t2}")
        # print(text_features)


    # loader = DataLoader(data, batch_size=batch_size, num_workers=data_workers, 
    #     shuffle=False, pin_memory=True, drop_last=True)



if __name__ == '__main__':
    main()
