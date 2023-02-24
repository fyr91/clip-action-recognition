# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2023-02-13 15:40:14
# @Last Modified by:   fyr91
# @Last Modified time: 2023-02-24 15:48:06
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

import time
import os

import clip
import torch
import torch.nn as nn

from action_config import *

from datasets import ActionFrames
from torch.utils.data import DataLoader
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_transform
from utils.Text_Prompt import text_prompt


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


# preset configs
num_seg = 8 # frames to process per seg
sim_header = "Transf" #Transf   meanP  LSTM  Conv_1D  Transf_cls
vit_bone = "weights/ViT-B-16.pt"
pretrain = "weights/vit-b-16-16f.pt"
batch_size = 1
data_workers = 1
input_size = 224
label_data = [*fall_desc,*fight_desc,*nudity_desc,
			  *fight_doc_desc,*suicide_desc,*climb_desc,
			  *gather_desc,*violence_desc,*exclusion_desc]

# target actions
fall_idx = [label_data.index(a) for a in fall_desc]
fight_idx = [label_data.index(a) for a in fight_desc]
nutidy_idx = [label_data.index(a) for a in nudity_desc]
fight_doc_idx = [label_data.index(a) for a in fight_doc_desc]
suicide_idx = [label_data.index(a) for a in suicide_desc]
climb_idx = [label_data.index(a) for a in climb_desc]
gathering_idx = [label_data.index(a) for a in gather_desc]
violence_idx = [label_data.index(a) for a in violence_desc]
excluded_idx = [label_data.index(a) for a in exclusion_desc]

# in seq fall, fight, nude, fight-doc, suicide, climb, gather, violence
threshs = {
	"falling":0.17, 
	"fighting":0.65, 
	"nudity":0.9,
	"fighting_doc":0.5, 
	"suicide":0.76, 
	"climbing":0.9, 
	"gathering":0.71, 
	"violence":0.6
	}


app = FastAPI(
	title = 'action detection server',
	description= 'detect actions from video segments',
	version= 'v0.2.4',
)


app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_methods=['*'],
	allow_headers=['*'],
	allow_credentials=['*'],
)


@app.on_event("startup")
async def startup_event():
	app.device = "cuda" if torch.cuda.is_available() else "cpu"
	app.model, clip_state_dict = clip.load(vit_bone, device=app.device, jit=False, 
		T=num_seg, tsm=False, dropout=0.0, emb_dropout=0.0)

	app.fusion_model = visual_prompt(sim_header, clip_state_dict, num_seg)
	app.model_text = TextCLIP(app.model)
	app.model_image = ImageCLIP(app.model)

	app.model_text = torch.nn.DataParallel(app.model_text).cuda()
	app.model_image = torch.nn.DataParallel(app.model_image).cuda()
	app.fusion_model = torch.nn.DataParallel(app.fusion_model).cuda()

	# half precision
	clip.model.convert_weights(app.model_text)  
	clip.model.convert_weights(app.model_image)

	print("INFO:     loading checkpoint")
	checkpoint = torch.load(pretrain)
	app.model.load_state_dict(checkpoint['model_state_dict'])
	app.fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])

	# evaluaiton mode
	app.model.eval()
	app.fusion_model.eval()

	print("INFO:     loading action labels")
	# data = Action_DATASETS(file_list, transform_data)
	classes, app.num_text_aug, text_dict = text_prompt(label_data)
	print(classes.shape)
	print(f"INFO:     {app.num_text_aug} labels generated")

	print("INFO:     generating label features")
	with torch.no_grad():
		t0 = time.time()
		text_inputs = classes.to(app.device)
		app.text_features = app.model.encode_text(text_inputs)
		app.text_features /= app.text_features.norm(dim=-1, keepdim=True)
		t1 = time.time()
		print(f"INFO:     text features generated in {t1-t0}")


@app.post("/cls")
async def cls(segs: List[str] = Form()):
	# print(segs)
	segs = segs[0].split(',')
	data_transform = get_transform(input_size)
	data = ActionFrames([segs], data_transform)
	loader = DataLoader(data, batch_size=batch_size, num_workers=data_workers, 
		shuffle=False, pin_memory=True, drop_last=False)
	print("INFO:     created data loader")

	res = {}
	event_idx = []
	frames = []
	events = []
	highest = {
		"falling":-1, 
		"fighting":-1, 
		"nudity":-1, 
		"fighting_doc":-1, 
		"suicide":-1, 
		"climbing":-1, 
		"gathering":-1, 
		"violence":-1
	}
	action_details = {}
	for label in label_data:
		action_details[label] = -1

	with torch.no_grad():
		for ii, image in enumerate(loader):
			t0 = time.time()
			image = image.view((-1, num_seg, 3) + image.size()[-2:])
			b, t, c, h, w = image.size()
			print(image.size())
			image_input = image.to(app.device).view(-1, c, h, w)
			image_features = app.model.encode_image(image_input).view(b, t, -1)
			image_features = app.fusion_model(image_features)
			image_features /= image_features.norm(dim=-1, keepdim=True)

			similarity = (100.0 * image_features @ app.text_features.T)
			similarity = similarity.view(b, app.num_text_aug, -1).softmax(dim=-1)
			similarity,_ = torch.max(similarity , dim=1)
			values_1, indices_1 = similarity.topk(1, dim=-1)
			values_5, indices_5 = similarity.topk(5, dim=-1)

			t1 = time.time()
			print("INFO:     =================================")
			for i in range(b):
				res[i] = []
				for j in range(5):
					temp = {}
					if action_details[label_data[indices_5[i][j]]] < values_5[i][j].item():
						action_details[label_data[indices_5[i][j]]] = values_5[i][j].item()

					if indices_5[i][j] in fall_idx:
						if highest["falling"] < values_5[i][j].item():
							highest["falling"] = values_5[i][j].item()
						if values_5[i][j] > threshs["falling"]: 
							temp = {"event":"falling", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in fight_idx:
						if highest["fighting"] < values_5[i][j].item():
							highest["fighting"] = values_5[i][j].item()
						if values_5[i][j] > threshs["fighting"]:
							temp = {"event":"fighting", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in nutidy_idx:
						if highest["nudity"] < values_5[i][j].item():
							highest["nudity"] = values_5[i][j].item()
						if values_5[i][j] > threshs["nudity"]:
							temp = {"event":"nudity", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in fight_doc_idx:
						if highest["fighting_doc"] < values_5[i][j].item():
							highest["fighting_doc"] = values_5[i][j].item()
						if values_5[i][j] > threshs["fighting_doc"]:
							temp = {"event":"fighting_doc", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in suicide_idx:
						if highest["suicide"] < values_5[i][j].item():
							highest["suicide"] = values_5[i][j].item()
						if values_5[i][j] > threshs["suicide"]:
							temp = {"event":"suicide", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in climb_idx:
						if highest["climbing"] < values_5[i][j].item():
							highest["climbing"] = values_5[i][j].item()
						if values_5[i][j] > threshs["climbing"]:
							temp = {"event":"climbing", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in gathering_idx:
						if highest["gathering"] < values_5[i][j].item():
							highest["gathering"] = values_5[i][j].item()
						if values_5[i][j] > threshs["gathering"]:
							temp = {"event":"gathering", "score":values_5[i][j].item()}
							res[i].append(temp)
					elif indices_5[i][j] in violence_idx:
						if highest["violence"] < values_5[i][j].item():
							highest["violence"] = values_5[i][j].item()
						if values_5[i][j] > threshs["violence"]:
							temp = {"event":"violence", "score":values_5[i][j].item()}
							res[i].append(temp)

		for detail_key in action_details:
			if action_details[detail_key] != -1:
				print(f"INFO:     actions - {detail_key} - {round(action_details[detail_key], 4)}")

		# print(f"video seg features generated in {t1-t0}")
		print("INFO:     =================================")
		for item_key in highest:
			if highest[item_key] != -1:
				print(f"INFO:     {item_key}: {round(highest[item_key], 4)}")
		print("INFO:     =================================")
	# check what event:
	print(f"INFO:     res - {res}")
	return res
