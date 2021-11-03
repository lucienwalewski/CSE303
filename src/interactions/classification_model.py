import os
import sys
import json
import subprocess
import numpy as np
import argparse
import torch
from torch import nn

# from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from resnet import ResNet


class NewModel(nn.Module):
    def __init__(self, resnet1, resnet2) -> None:
        super(NewModel).__init__()
        self.resnet1 = resnet1
        self.resnet2 = resnet2
        self.classification_layer = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.Linear(512, 2) # Number of outputs?
        )

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classification_layer(torch.relu(x))
        return x

    def train():
        # FIXME


def generate_new_model(resnet):
    two_stream_model = NewModel(resnet, resnet)


def create_two_stream_model():
	pass

if __name__ == '__main__':

    opt = argparse.ArgumentParser()
    opt.mean = get_mean()
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    # Correct configuration of opt to be done
    # FIXME

    resnet = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    model = generate_new_model(resnet)



#     opt = parse_opts()
#     opt.mean = get_mean()
#     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
#     opt.sample_size = 112
#     opt.sample_duration = 16
#     opt.n_classes = 400

#     model = generate_model(opt)
#     print('loading model {}'.format(opt.model))
#     model_data = torch.load(opt.model)
#     assert opt.arch == model_data['arch']
#     model.load_state_dict(model_data['state_dict'])
#     model.eval()
#     if opt.verbose:
#         print(model)

#     input_files = []
#     with open(opt.input, 'r') as f:
#         for row in f:
#             input_files.append(row[:-1])

#     class_names = []
#     with open('class_names_list') as f:
#         for row in f:
#             class_names.append(row[:-1])

#     ffmpeg_loglevel = 'quiet'
#     if opt.verbose:
#         ffmpeg_loglevel = 'info'

#     if os.path.exists('tmp'):
#         subprocess.call('rm -rf tmp', shell=True)

#     outputs = []
#     for input_file in input_files:
#         video_path = os.path.join(opt.video_root, input_file)
#         if os.path.exists(video_path):
#             print(video_path)
#             subprocess.call('mkdir tmp', shell=True)
#             subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
#                             shell=True)

#             result = classify_video('tmp', input_file, class_names, model, opt)
#             outputs.append(result)

#             subprocess.call('rm -rf tmp', shell=True)
#         else:
#             print('{} does not exist'.format(input_file))

#     if os.path.exists('tmp'):
#         subprocess.call('rm -rf tmp', shell=True)

#     with open(opt.output, 'w') as f:
#         json.dump(outputs, f)