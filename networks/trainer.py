import math
import os
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn import init
import cv2
import torchvision.transforms as transforms
import skimage.feature
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from skimage.color import rgb2hsv
from torchvision.transforms import functional as TF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

class SA_layer(nn.Module):
    def __init__(self, dim=128, head_size=4):
        super(SA_layer, self).__init__()
        self.mha = nn.MultiheadAttention(dim, head_size)
        self.ln1 = nn.LayerNorm(dim) 
        self.fc1 = nn.Linear(dim, dim)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim) 
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim = x.shape
        x = torch.transpose(x, 1, 0)
        y, _ = self.mha(x, x, x)
        x = self.ln1(x + y)
        x = torch.transpose(x, 1, 0)
        x = x.reshape(batch_size * len_size, fea_dim)
        x = x + self.fc2(self.ac(self.fc1(x)))
        x = x.reshape(batch_size, len_size, fea_dim)
        x = self.ln2(x)
        return x

import math
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
class Attention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out

class COOI():
    def __init__(self):
        self.stride = 32
        self.cropped_size = 224
        self.score_filter_size_list = [[3, 3], [2, 2]]
        self.score_filter_num_list = [2, 2]
        self.score_nms_size_list = [[3, 3], [3, 3]]
        self.score_nms_padding_list = [[1, 1], [1, 1]]
        self.score_corresponding_patch_size_list = [[224, 224], [112, 112]]
        self.score_filter_type_size = len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        device = fm.device  
        self.Attention=Attention(channel=2048).to(device)
        with torch.no_grad():
            batch_size, _, fm_height, fm_width = fm.size()
            scale_min = torch.min(scale, axis=1, keepdim=True)[0].long()#scale_min=(64,1)
            scale_base = (scale - scale_min).long() // 2  #scale_base=(64,2)# torch.div(scale-scale_min,2,rounding_mode='floor')算中心位置
            input_loc_list = []
            fm = self.Attention(fm)

            for type_no in range(self.score_filter_type_size):
                score_avg = nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no],
                                                     stride=1)  # (7,2048,5,5), (7,2048,6,6)
                score_sum = torch.sum(score_avg, dim=1,
                                      keepdim=True)  
                _, _, score_height, score_width = score_sum.size()
                patch_height, patch_width = self.score_corresponding_patch_size_list[type_no]
                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat = score_sum.view(batch_size, -1)
                    value_max, loc_max_flat = torch.max(score_sum_flat, dim=1)
                    # loc_max=torch.stack((torch.div(loc_max_flat,score_width,rounding_mode='floor'), loc_max_flat%score_width), dim=1)
                    loc_max = torch.stack((loc_max_flat // score_width, loc_max_flat % score_width), dim=1)
     
                    top_patch = nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1,
                                                         padding=self.score_nms_padding_list[type_no])
                    value_max = value_max.view(-1, 1, 1, 1)
                    erase = (
                                top_patch != value_max).float()  
                    score_sum = score_sum * erase
                    loc_rate_h = (2 * loc_max[:, 0] + fm_height - score_height + 1) / (2 * fm_height)
                    loc_rate_w = (2 * loc_max[:, 1] + fm_width - score_width + 1) / (2 * fm_width)
                    loc_rate = torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center = (scale_base + scale_min * loc_rate).long()
                    loc_top = loc_center[:, 0] - patch_height // 2
                    loc_bot = loc_center[:, 0] + patch_height // 2 + patch_height % 2
                    loc_lef = loc_center[:, 1] - patch_width // 2
                    loc_rig = loc_center[:, 1] + patch_width // 2 + patch_width % 2
                    loc_tl = torch.stack((loc_top, loc_lef), dim=1)
                    loc_br = torch.stack((loc_bot, loc_rig), dim=1)
                    loc_below = loc_tl.detach().clone() 
                    loc_below[loc_below > 0] = 0
                    loc_br -= loc_below
                    loc_tl -= loc_below
                    loc_over = loc_br - scale.long() 
                    loc_over[loc_over < 0] = 0
                    loc_tl -= loc_over
                    loc_br -= loc_over
                    loc_tl[loc_tl < 0] = 0 
                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor = torch.stack(input_loc_list, dim=1) 
            return input_loc_tensor


class Patch5Model(nn.Module):
    def __init__(self):
        super(Patch5Model, self).__init__()
        self.resnet = resnet50(pretrained=True)  # debug
        self.COOI = COOI()
        self.mha_list = nn.Sequential(
            SA_layer(128, 4),
            SA_layer(128, 4),
            SA_layer(128, 4)
        )
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, 1)
    def extract_h_channel(self, images):
        images_np = images.permute(0, 2, 3, 1).cpu().numpy() 
        hsv_images = rgb2hsv(images_np)
        h_channel = hsv_images[..., 0]
        return torch.tensor(h_channel).unsqueeze(1).float() 

    def compute_glcm_features(self, h_channel):
        properties = ['dissimilarity','homogeneity', 'ASM']
        distances = [2]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm_matrices = []

        for img in h_channel:
            img_np = img.squeeze(0).cpu().numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            if img_np.max() == img_np.min():
                glcm_features = np.zeros((1, len(properties) * len(distances) * len(angles)))
            else:
                glcm = graycomatrix(img_np, distances, angles, symmetric=True, normed=True)
                glcm_props = []
                for prop in properties:
                    glcm_props.append(graycoprops(glcm, prop).flatten())
                glcm_features = np.hstack(glcm_props)
            glcm_matrices.append(glcm_features)

        return np.array(glcm_matrices)

    def reduce_glcm_with_pca(self, glcm_features):
        if np.isnan(glcm_features).any() or np.isinf(glcm_features).any():
            glcm_features = np.nan_to_num(glcm_features, nan=0.0, posinf=1e6, neginf=-1e6)

        variances = np.var(glcm_features, axis=0)
        if np.any(variances == 0):
            print("Warning: Some features have zero variance. Removing these features.")
            glcm_features = glcm_features[:, variances != 0]

        if glcm_features.shape[1] == 0:
            print("All features have zero variance. Returning a zero tensor.")
            batch_size = glcm_features.shape[0]
            pca_features = np.zeros((batch_size, 3, 224, 224))
            return torch.tensor(pca_features).float()

        glcm_features = StandardScaler().fit_transform(glcm_features)

        if np.isnan(glcm_features).any() or np.isinf(glcm_features).any():
            glcm_features = np.nan_to_num(glcm_features, nan=0.0, posinf=1e6, neginf=-1e6)

        pca = PCA(n_components=3)
        try:
            pca_features = pca.fit_transform(glcm_features)
        except ValueError as e:
            print(f"Error during PCA: {e}")
            batch_size = glcm_features.shape[0]
            pca_features = np.zeros((batch_size, 3))

        if np.isnan(pca_features).any() or np.isinf(pca_features).any():
            print("PCA features contain NaN or Inf values.")
            pca_features = np.nan_to_num(pca_features, nan=0.0, posinf=1e6, neginf=-1e6)

        batch_size = glcm_features.shape[0]
        pca_features = pca_features.reshape(batch_size, 3)
        pca_features_expanded = np.tile(pca_features[:, :, np.newaxis, np.newaxis], (1, 1, 224, 224))
        return torch.tensor(pca_features_expanded).float()

    def forward(self, input_img, cropped_img, scale):
        x = cropped_img
        batch_size, p, _, _ = x.shape  # [batch_size, 3, 224, 224]
        h_channel = self.extract_h_channel(x)
        glcm_features = self.compute_glcm_features(h_channel)
        #print(f"glcm_features shape: {glcm_features.shape}")
        pca_features = self.reduce_glcm_with_pca(glcm_features).to(x.device)
        #print(f"pca_features shape: {pca_features.shape}")
        _,glcm_embeddings = self.resnet(pca_features.detach())
        s_glcm_embedding = self.ac(self.fc1(glcm_embeddings))
        s_glcm_embedding = s_glcm_embedding.view(-1, 1, 128)

        fm, whole_embedding = self.resnet(x)  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print(whole_embedding.shape)
        # print(fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)
        # print(s_whole_embedding.shape)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)
        _, proposal_size, _ = input_loc.size()
        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device) 

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                img_patch = input_img[batch_no][:, t:b, l:r]
                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()
                if patch_height == 224 and patch_width == 224:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no + 1] = F.interpolate(img_patch[None, ...],
                                                                                       size=(224, 224),
                                                                                       mode='bilinear',
                                                                                       align_corners=True)   # [N, 6, 3, 224, 224]
        #print(window_imgs.shape)
        # exit()
        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224) 
        _, window_embeddings = self.resnet(window_imgs.detach())  
        s_window_embedding = self.ac(self.fc1(window_embeddings)) 
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)
        # print(s_window_embedding.shape)
        all_embeddings = torch.cat((s_window_embedding, s_whole_embedding, s_glcm_embedding), 1)
        #print(all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])

        return all_logits

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model=Patch5Model()
            if torch.cuda.device_count()>1:
                 self.model=nn.DataParallel(self.model)
         if not self.isTrain or opt.continue_train:
             #self.model = resnet50(num_classes=1)
             self.model=Patch5Model()
             if torch.cuda.device_count()>1:
                 self.model=nn.DataParallel(self.model)
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        if len(opt.gpu_ids)==0:
            self.model.to('cpu')
        else:
            self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img = data[0] 
        self.cropped_img = data[1].to(self.device)
        self.label = data[2].to(self.device).float() 

    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
