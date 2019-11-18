"""
Model script for ZSG Network, adapted from: https://github.com/TheShadow29/zsgnet-pytorch
Author: Arka Sadhu

SSD-VGG implementation adapted from the repository: https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py

FPN-ResNet taken from the repository: https://github.com/yhenon/pytorch-retinanet/blob/master/model.py
"""
import os
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torchvision.models as tvm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from fpn_resnet import FPN_backbone
from utils.anchors import create_grid
from typing import Dict, Any

device='cpu'
if torch.cuda.is_available():
  device='cuda'

# conv2d, conv2d_relu are adapted from
# https://github.com/fastai/fastai/blob/5c4cefdeaf11fdbbdf876dbe37134c118dca03ad/fastai/layers.py#L98
def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1,
           padding: int = None, bias=False) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None:
        padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                     padding=padding, bias=bias)


def conv2d_relu(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None,
                bn: bool = False, bias: bool = False) -> nn.Sequential:
    """
    Create a `conv2d` layer with `nn.ReLU` activation
    and optional(`bn`) `nn.BatchNorm2d`: `ni` input, `nf` out
    filters, `ks` kernel, `stride`:stride, `padding`:padding,
    `bn`: batch normalization.
    """
    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class BackBone(nn.Module):
    """
    A general purpose Backbone class.
    For a new network, need to redefine:
    --> encode_feats
    Optionally after_init
    """

    def __init__(self, encoder: nn.Module, cfg: dict, out_chs=256):
        """
        Make required forward hooks
        """
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.cfg = cfg
        self.out_chs = out_chs
        self.after_init()

    def after_init(self):
        pass

    def num_channels(self):
        raise NotImplementedError

    def concat_we(self, x, we, only_we=False, only_grid=False):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W (one feature map)
        we: B x wdim (the language vector)
        Output: concatenated word embedding and grid centers
        """
        # Both cannot be true
        assert not (only_we and only_grid)

        # Create the grid
        grid = create_grid((x.size(2), x.size(3)),
                           flatten=False).to(self.device)
        grid = grid.permute(2, 0, 1).contiguous()

        # TODO: Slightly cleaner implementation?
        grid_tile = grid.view(
            1, grid.size(0), grid.size(1), grid.size(2)).expand(
            we.size(0), grid.size(0), grid.size(1), grid.size(2))

        # In case we only need the grid
        # Basically, don't use any image/language information
        if only_grid:
            return grid_tile

        # Expand word embeddings
        word_emb_tile = we.view(
            we.size(0), we.size(1), 1, 1).expand(
                we.size(0), we.size(1), x.size(2), x.size(3))

        # In case performing image blind (requiring only language)
        if only_we:
            return word_emb_tile

        # Concatenate along the channel dimension
        return torch.cat((x, word_emb_tile, grid_tile), dim=1)

    def encode_feats(self, inp):
        return self.encoder(inp)

    def forward(self, inp, we=None,
                only_we=False, only_grid=False):
        """
        expecting word embedding of shape B x WE.
        If only image features are needed, don't
        provide any word embedding
        """
        feats = self.encode_feats(inp)
        # If we want to do normalization of the features
        if self.cfg['do_norm']:
            feats = [
                feat / feat.norm(dim=1).unsqueeze(1).expand(*feat.shape)
                for feat in feats
            ]

        # For language blind setting, can directly return the features
        if we is None:
            return feats

        if self.cfg['do_norm']:
            b, wdim = we.shape
            we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)

        out = [self.concat_we(
            f, we, only_we=only_we, only_grid=only_grid) for f in feats]

        return out


class RetinaBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone(self.num_chs, self.cfg, feat_size=self.out_chs)

    def num_channels(self):
        return [self.encoder.layer2[-1].conv3.out_channels,
                self.encoder.layer3[-1].conv3.out_channels,
                self.encoder.layer4[-1].conv3.out_channels]

    def encode_feats(self, inp):
        x = self.encoder.conv1(inp)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        feats = self.fpn([x2, x3, x4])
        return feats


class SSDBackBone(BackBone):
    """
    ssd_vgg.py already implements encoder
    """

    def encode_feats(self, inp):
        return self.encoder(inp)


class ZSGNet(nn.Module):
    """
    The main model
    Uses SSD like architecture but for Lang+Vision
    """

    def __init__(self, backbone, n_anchors=1, final_bias=0., cfg=None):
        super().__init__()
        # assert isinstance(backbone, BackBone)
        self.backbone = backbone

        # Assume the output from each
        # component of backbone will have 256 channels
        self.device = torch.device(device)

        self.cfg = cfg

        # should be len(ratios) * len(scales)
        self.n_anchors = n_anchors

        self.emb_dim = cfg['emb_dim']
        self.bid = cfg['use_bidirectional']
        self.lstm_dim = cfg['lstm_dim']

        # Calculate output dimension of LSTM
        self.lstm_out_dim = self.lstm_dim * (self.bid + 1)

        # Separate cases for language, image blind settings
        if self.cfg['use_lang'] and self.cfg['use_img']:
            self.start_dim_head = self.lstm_dim*(self.bid+1) + 256 + 2
        elif self.cfg['use_img'] and not self.cfg['use_lang']:
            # language blind
            self.start_dim_head = 256
        elif self.cfg['use_lang'] and not self.cfg['use_img']:
            # image blind
            self.start_dim_head = self.lstm_dim*(self.bid+1)
        else:
            # both image, lang blind
            self.start_dim_head = 2

        # If shared heads for classification, box regression
        # This is the config used in the paper
        if self.cfg['use_same_atb']:
            bias = torch.zeros(5 * self.n_anchors)
            bias[torch.arange(4, 5 * self.n_anchors, 5)] = -4
            self.att_reg_box = self._head_subnet(
                5, self.n_anchors, final_bias=bias,
                start_dim_head=self.start_dim_head
            )
        # This is not used. Kept for historical purposes
        else:
            self.att_box = self._head_subnet(
                1, self.n_anchors, -4., start_dim_head=self.start_dim_head)
            self.reg_box = self._head_subnet(
                4, self.n_anchors, start_dim_head=self.start_dim_head)

        self.lstm = nn.LSTM(self.emb_dim, self.lstm_dim,
                            bidirectional=self.bid, batch_first=False)
        self.after_init()

    def after_init(self):
        "Placeholder if any child class needs something more"
        pass

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256,
                     start_dim_head=256):
        """
        Convenience function to create attention and regression heads
        """
        layers = [conv2d_relu(start_dim_head, chs, bias=True)]
        layers += [conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        return nn.Sequential(*layers)

    def permute_correctly(self, inp, outc):
        """
        Basically square box features are flattened
        """
        # inp is features
        # B x C x H x W -> B x H x W x C
        out = inp.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, outc)
        return out

    def concat_we(self, x, we, append_grid_centers=True):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W
        we: B x wdim
        """
        b, wdim = we.shape
        we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)
        word_emb_tile = we.view(we.size(0), we.size(1),
                                1, 1).expand(we.size(0),
                                             we.size(1),
                                             x.size(2), x.size(3))

        if append_grid_centers:
            grid = create_grid((x.size(2), x.size(3)),
                               flatten=False).to(self.device)
            grid = grid.permute(2, 0, 1).contiguous()
            grid_tile = grid.view(1, grid.size(0), grid.size(1), grid.size(2)).expand(
                we.size(0), grid.size(0), grid.size(1), grid.size(2))

            return torch.cat((x, word_emb_tile, grid_tile), dim=1)
        return torch.cat((x, word_emb_tile), dim=1)

    def lstm_init_hidden(self, bs):
        """
        Initialize the very first hidden state of LSTM
        Basically, the LSTM should be independent of this
        """
        if not self.bid:
            hidden_a = torch.randn(1, bs, self.lstm_dim)
            hidden_b = torch.randn(1, bs, self.lstm_dim)
        else:
            hidden_a = torch.randn(2, bs, self.lstm_dim)
            hidden_b = torch.randn(2, bs, self.lstm_dim)

        hidden_a = hidden_a.to(self.device)
        hidden_b = hidden_b.to(self.device)

        return (hidden_a, hidden_b)

    def apply_lstm(self, word_embs, qlens, max_qlen, get_full_seq=False):
        """
        Applies lstm function.
        word_embs: word embeddings, B x seq_len x 300
        qlen: length of the phrases
        Try not to fiddle with this function.
        IT JUST WORKS
        """
        # B x T x E
        bs, max_seq_len, emb_dim = word_embs.shape
        # bid x B x L
        self.hidden = self.lstm_init_hidden(bs)
        # B x 1, B x 1
        qlens1, perm_idx = qlens.sort(0, descending=True)
        # B x T x E (permuted)
        qtoks = word_embs[perm_idx]
        # T x B x E
        embeds = qtoks.permute(1, 0, 2).contiguous()
        # Packed Embeddings
        packed_embed_inp = pack_padded_sequence(
            embeds, lengths=qlens1, batch_first=False)
        # To ensure no pains with DataParallel
        # self.lstm.flatten_parameters()
        lstm_out1, (self.hidden, _) = self.lstm(packed_embed_inp, self.hidden)

        # T x B x L
        lstm_out, req_lens = pad_packed_sequence(
            lstm_out1, batch_first=False, total_length=max_qlen)

        # TODO: Simplify getting the last vector
        masks = (qlens1-1).view(1, -1, 1).expand(max_qlen,
                                                 lstm_out.size(1), lstm_out.size(2))
        qvec_sorted = lstm_out.gather(0, masks.long())[0]

        qvec_out = word_embs.new_zeros(qvec_sorted.shape)
        qvec_out[perm_idx] = qvec_sorted
        # if full sequence is needed for future work
        if get_full_seq:
            lstm_out_1 = lstm_out.transpose(1, 0).contiguous()
            return lstm_out_1
        return qvec_out.contiguous()

    def forward(self, inp: Dict[str, Any]):
        """
        Forward method of the model
        inp0 : image to be used
        inp1 : word embeddings, B x seq_len x 300
        qlens: length of phrases

        The following is performed:
        1. Get final hidden state features of lstm
        2. Get image feature maps
        3. Concatenate the two, specifically, copy lang features
        and append it to all the image feature maps, also append the
        grid centers.
        4. Use the classification, regression head on this concatenated features
        The matching with groundtruth is done in loss function and evaluation
        """
        inp0 = inp['img']
        inp1 = inp['qvec']
        qlens = inp['qlens']
        max_qlen = int(qlens.max().item())
        req_embs = inp1[:, :max_qlen, :].contiguous()

        req_emb = self.apply_lstm(req_embs, qlens, max_qlen)

        # image blind
        if self.cfg['use_lang'] and not self.cfg['use_img']:
            # feat_out = self.backbone(inp0)
            feat_out = self.backbone(inp0, req_emb, only_we=True)

        # language blind
        elif self.cfg['use_img'] and not self.cfg['use_lang']:
            feat_out = self.backbone(inp0)

        elif not self.cfg['use_img'] and not self.cfg['use_lang']:
            feat_out = self.backbone(inp0, req_emb, only_grid=True)
        # see full language + image (happens by default)
        else:
            feat_out = self.backbone(inp0, req_emb)

        # Strategy depending on shared head or not
        if self.cfg['use_same_atb']:
            att_bbx_out = torch.cat([self.permute_correctly(
                self.att_reg_box(feature), 5) for feature in feat_out], dim=1)
            att_out = att_bbx_out[..., [-1]]
            bbx_out = att_bbx_out[..., :-1]
        else:
            att_out = torch.cat(
                [self.permute_correctly(self.att_box(feature), 1)
                 for feature in feat_out], dim=1)
            bbx_out = torch.cat(
                [self.permute_correctly(self.reg_box(feature), 4)
                 for feature in feat_out], dim=1)

        feat_sizes = torch.tensor([[f.size(2), f.size(3)]
                                   for f in feat_out]).to(self.device)

        # Used mainly due to dataparallel consistency
        num_f_out = torch.tensor([len(feat_out)]).to(self.device)

        out_dict = {}
        out_dict['att_out'] = att_out
        out_dict['bbx_out'] = bbx_out
        out_dict['feat_sizes'] = feat_sizes
        out_dict['num_f_out'] = num_f_out

        return out_dict


def get_default_net(num_anchors=1, cfg=None):
    """
    Constructs the network based on the given config
    """
    if cfg['mdl_to_use'] == 'retina':
        encoder = tvm.resnet50(True)
        backbone = RetinaBackBone(encoder, cfg)
    elif cfg['mdl_to_use'] == 'ssd_vgg':
        encoder = ssd_vgg.build_ssd('train', cfg=cfg)
        encoder.vgg.load_state_dict(
            torch.load('./weights/vgg16_reducedfc.pth'))
        print('loaded pretrained vgg backbone')
        backbone = SSDBackBone(encoder, cfg)
        # backbone = encoder

    zsg_net = ZSGNet(backbone, num_anchors, cfg=cfg)
    return zsg_net


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfg=None):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # self.cfg = (coco, voc)[num_classes == 21]
        # self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.cfg = cfg

        # SSD network
        self.vgg = nn.ModuleList(base)
        # self.vgg = tvm.vgg16(pretrained=True)
        # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        self.fproj1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fproj2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.fproj3 = nn.Conv2d(512, 256, kernel_size=1)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # s = self.L2Norm(x)
        s = x / x.norm(dim=1, keepdim=True)
        sources.append(s)
        # print(f'Adding1 of dim {s.shape}')

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        # print(f'Adding2 of dim {x.shape}')

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                # print(f'Adding3 of dim {x.shape}')

        out_sources = [self.fproj1(sources[0]), self.fproj2(
            sources[1]), self.fproj3(sources[2])] + sources[3:]
        if self.cfg['resize_img'][0] >= 600:
            # To Reduce the computation
            return out_sources[1:]
        return out_sources

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21, cfg=None):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes, cfg=cfg)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    standard Basic block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def pad_out(k):
    "padding to have same size"
    return (k-1)//2


class FPN_backbone(nn.Module):
    """
    A different fpn, doubt it will work
    """

    def __init__(self, inch_list, cfg, feat_size=256):
        super().__init__()

#         self.backbone = backbone

        # expects c3, c4, c5 channel dims
        self.inch_list = inch_list
        self.cfg = cfg
        c3_ch, c4_ch, c5_ch = self.inch_list
        self.feat_size = feat_size

        self.P7_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, stride=2,
                              kernel_size=3,
                              padding=1)
        self.P6 = nn.Conv2d(in_channels=c5_ch,
                            out_channels=self.feat_size,
                            kernel_size=3, stride=2, padding=pad_out(3))
        self.P5_1 = nn.Conv2d(in_channels=c5_ch,
                              out_channels=self.feat_size,
                              kernel_size=1, padding=pad_out(1))

        self.P5_2 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size,
                              kernel_size=3, padding=pad_out(3))

        self.P4_1 = nn.Conv2d(in_channels=c4_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P4_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

        self.P3_1 = nn.Conv2d(in_channels=c3_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P3_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

    def forward(self, inp):
        # expects inp to be output of c3, c4, c5
        c3, c4, c5 = inp
        p51 = self.P5_1(c5)
        p5_out = self.P5_2(p51)

        # p5_up = F.interpolate(p51, scale_factor=2)
        p5_up = F.interpolate(p51, size=(c4.size(2), c4.size(3)))
        p41 = self.P4_1(c4) + p5_up
        p4_out = self.P4_2(p41)

        # p4_up = F.interpolate(p41, scale_factor=2)
        p4_up = F.interpolate(p41, size=(c3.size(2), c3.size(3)))
        p31 = self.P3_1(c3) + p4_up
        p3_out = self.P3_2(p31)

        p6_out = self.P6(c5)

        p7_out = self.P7_2(F.relu(p6_out))
        if self.cfg['resize_img'] == [600, 600]:
            return [p4_out, p5_out, p6_out, p7_out]

        # p8_out = self.p8_gen(F.relu(p7_out))
        p8_out = F.adaptive_avg_pool2d(p7_out, 1)
        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]


class PyramidFeatures(nn.Module):
    """
    Pyramid Features, especially for Resnet
    """

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size,
                            kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        """
        Inputs should be from layer2,3,4
        """
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        P6_x = self.P6(C5)
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class ResNet(nn.Module):
    """
    Basic Resnet Module
    """

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2] -
                                                                                  1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2] -
                                                                                  1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.freeze_bn()
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Convenience function to generate layers given blocks and
        channel dimensions
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        """
        inputs should be images
        """
        img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        return features


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50'], model_dir='.'), strict=False)
    return model
