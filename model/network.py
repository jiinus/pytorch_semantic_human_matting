
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.M_Net import M_net
from model.T_Net_psp import PSPNet


def t_to_m(bg, unsure, fg, img, alpha, back, front):
    a = random.random() * 3
    if a < 1:
        patch_size = 320
    elif a < 2:
        patch_size = 480
    else:
        patch_size = 640

    b = random.random()
    if b < 0.5:
        bg = torch.flip(bg, (-1,))
        unsure = torch.flip(unsure, (-1,))
        fg = torch.flip(fg, (-1,))
        img = torch.flip(img, (-1,))
        alpha = torch.flip(alpha, (-1,))
        back = torch.flip(back, (-1,))
        front = torch.flip(front, (-1,))

    x = random.randint(0, 800 - patch_size)
    y = random.randint(0, 800 - patch_size)
    x_end = x + patch_size
    y_end = y + patch_size

    bg = F.interpolate(bg[:,:, x:x_end, y:y_end], size=(320, 320))
    unsure = F.interpolate(unsure[:,:, x:x_end, y:y_end], size=(320, 320))
    fg = F.interpolate(fg[:,:, x:x_end, y:y_end], size=(320, 320))
    img = F.interpolate(img[:,:, x:x_end, y:y_end], size=(320, 320))
    alpha = F.interpolate(alpha[:,:, x:x_end, y:y_end], size=(320, 320))
    back = F.interpolate(back[:,:, x:x_end, y:y_end], size=(320, 320))
    front = F.interpolate(front[:,:, x:x_end, y:y_end], size=(320, 320))

    return bg, unsure, fg, img, alpha, back, front
    
    
class net_T(nn.Module):
    # Train T_net
    def __init__(self):

        super(net_T, self).__init__()

        self.t_net = PSPNet()

    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        return trimap

class net_M(nn.Module):
    '''
		train M_net
    '''

    def __init__(self):

        super(net_M, self).__init__()
        self.m_net = M_net()

    def forward(self, input, trimap):

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return alpha_p

class net_F(nn.Module):
    '''
		end to end net 
    '''

    def __init__(self):

        super(net_F, self).__init__()

        self.t_net = PSPNet()
        self.m_net = M_net()



    def forward(self, input):

    	# trimap
        trimap = self.t_net(input['img'])
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        bg, unsure, fg = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        bg, unsure, fg, img, alpha_g, back, front = t_to_m(bg, unsure, fg, input['img'], input['alpha_g'], input['back'], input['front'])
        m_net_input = torch.cat((input['img'], bg, unsure, fg), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p, img, alpha_g, back, front
