import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import  numpy as np
import kornia
import torch.nn.functional as F

def loss_builder1():
    criterion_1_1 = nn.NLLLoss(ignore_index=255)
    criterion_1_2 = DiceLoss(class_num=4)
    criterion = [criterion_1_1,criterion_1_2]
    return criterion


####################### New Losses

#Focal Loss
class FocalLoss(nn.Module):#Focal Loss
    def __init__(self):#0.4: less important than original loss
        super(FocalLoss, self).__init__()
        

    def forward(self, outputs, targets):#, log_sigma_ce, log_sigma_aux):#, log_sigma_hd):
        kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        loss =  kornia.losses.FocalLoss(**kwargs)(outputs,targets).cuda()
        return loss


import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg

softmax_helper = lambda x: F.softmax(x, 1)

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # channel
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf

#Boundary Loss
class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
            gt_sdf = compute_sdf(y_onehot.cpu().numpy(), net_output.shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != net_output.device:
            phi = phi.to(net_output.device).type(torch.float32)
        # pred = net_output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxy,bcxy->bcxy", net_output[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss
    
    
 #ref: https://github.com/JunMa11/SegLoss/blob/71b14900e91ea9405d9705c95b451fc819f24c70/test/loss_functions/boundary_loss.py#L131
#from nnunet.training.loss_functions.TopK_loss import TopKLoss
#from nnunet.utilities.nd_softmax import softmax_helper
#from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
#from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg
import numpy as np

import numpy as np
import torch
from torch import nn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape
    
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
            
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

##Soft Dice Loss
class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.,
                 square=False):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc




#Generalized Dice Loss + Boundary Loss
class DC_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super(DC_and_BD_loss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.bd = BDLoss(**bd_kwargs)
        

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target)
        if self.aggregate == "sum":
            alpha = 0.6
            result = alpha*dc_loss + (1-alpha)*bd_loss
        else:
            raise NotImplementedError("nah son") 
        return result       

class DC_and_FC_loss(nn.Module):
    def __init__(self, kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}):
        super(DC_and_FC_loss, self).__init__()
        kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        self.fc =  kornia.losses.FocalLoss(**kwargs)
        self.dc = SoftDiceLoss()
    def forward(self, net_output, target):
        fc_loss = self.fc(net_output, target)
        dc_loss = self.dc(net_output, target)
        alpha = 0.6
        return (1-alpha)*fc_loss + alpha*dc_loss

class F_CE(nn.Module):
    def __init__(self,t = 0):
        super(F_CE, self).__init__()
        self.loss = F.cross_entropy()
        self.t = 0
    def forward(self, net_output, target):
        loss = self.loss(net_output, target, reduction = 'mean')
        return loss
    
class PSPLoss(nn.Module):#PSPLoss = original loss + aux_loss
    def __init__(self, aux_weight = 0.4):#0.4: less important than original loss
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
    def forward(self, outputs, targets):#, log_sigma_hd):
        loss = F.cross_entropy(outputs[0], targets, reduction = 'mean').cuda()
        loss_aux = F.cross_entropy(outputs[1], targets, reduction = 'mean').cuda()
        #bd_loss = BDLoss()(outputs[0],targets).cuda()
        #print(f"loss: {loss} loss_aux: {loss_aux} bd_loss: {bd_loss}")
        return loss + self.aux_weight*loss_aux

    
def loss_builder2():
    criterion = PSPLoss(0.3)
    return criterion
    
def ch_loss_builder2():
    criterion = SoftDiceLoss()
#     criterion = DC_and_FC_loss()
    #criterion = DC_and_BD_loss(soft_dice_kwargs= {'batch_dice' : True, 'do_bg' : False, 'smooth' : 1e-5, 'square' : False}, bd_kwargs = {})#(outputs[0],targets).cuda()
    #criterion = F.cross_entropy()
    return criterion

def ch_loss_builder3():
    criterion = SoftDiceLoss()
    return criterion

