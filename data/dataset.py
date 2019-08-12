
import cv2
import os
import random
import numpy as np
import math

from data import gen_trimap

import torch
import torch.utils.data as data

def composite(image, a, bg):
    fg = np.array(image, np.float32)
    alpha = np.expand_dims(a/255., axis=2).astype(np.float32)
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)

    # we need bg for calculating compositional loss in M_net
    return comp

## crop & resize bg so that img and bg have same size
def resize_bg(bg, h, w):
    bh, bw = bg.shape[:2]
    wratio, hratio = w/bw, h/bh
    ratio = wratio if wratio > hratio else hratio
    bg = cv2.resize(bg, (math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)
    bh, bw = bg.shape[:2]
    assert bh>=h and bw>=w
    bg = np.array(bg[(bh - h) // 2: (bh - h) // 2 + h, (bw - w) // 2: (bw - w) // 2 + w], np.float32)
    assert h, w == bg.shape[:2]

    return bg  # attention: bg is in float32

# randomly crop images, then resize to patch_size
def random_patch(image, trimap, patch_size, bg=None, a=None):
    h, w = image.shape[:2]
    max_size = max(h, w)
    min_size = max_size // 2

    count=0
    while True:
        sqr_tri = np.zeros((max_size, max_size), np.uint8)
        if isinstance(bg, np.ndarray):
            sqr_img = np.zeros((max_size, max_size, 4), np.uint8)
            sqr_bga = np.zeros((max_size, max_size, 3), np.uint8)
            bga = resize_bg(bg, h, w)
        elif isinstance(a, np.ndarray):
            sqr_img = np.zeros((max_size, max_size, 3), np.uint8)
            sqr_bga = np.zeros((max_size, max_size), np.uint8)
            bga = a
        else:
            raise ValueError('no bg or alpha given from input!')

        if h >= w:
            sqr_tri[:, (h - w) // 2: (h - w) // 2 + w] = trimap
            sqr_img[:, (h - w) // 2: (h - w) // 2 + w] = image
            sqr_bga[:, (h - w) // 2: (h - w) // 2 + w] = bga
        else:
            sqr_tri[(w - h) // 2: (w - h) // 2 + h, :] = trimap
            sqr_img[(w - h) // 2: (w - h) // 2 + h, :] = image
            sqr_bga[(w - h) // 2: (w - h) // 2 + h, :] = bga

        crop_size = random.randint(min_size, max_size)  # both value are inclusive
        x = random.randint(0, max_size - crop_size)  # 0 is inclusive
        y = random.randint(0, max_size - crop_size)
        trimap_temp = sqr_tri[y: y + crop_size, x: x + crop_size]
        if len(np.where(trimap_temp == 128)[0]) > 0:  # check if unknown area is included
            image = sqr_img[y: y + crop_size, x: x + crop_size]
            bga = sqr_bga[y: y + crop_size, x: x + crop_size]
            break
        elif len(np.where(trimap == 128)[0]) == 0:
            print('Warning: No unknown area in current trimap! Refer to <exception> folder.')
            os.makedirs('ckpt/exceptions', exist_ok=True)
            cv2.imwrite('ckpt/exceptions/img_{}_{}.png'.format(str(h), str(w)), image)
            cv2.imwrite('ckpt/exceptions/tri_{}_{}.png'.format(str(h), str(w)), trimap)
        elif count > 3:
            print('Warning & Error: cannot find right patch randomly, use max_size instead! Refer to <exceptions> folder.')
            os.makedirs('ckpt/exceptions', exist_ok=True)
            cv2.imwrite('ckpt/exceptions/img_{}_{}.png'.format(str(h), str(w)), image)
            cv2.imwrite('ckpt/exceptions/tri_{}_{}.png'.format(str(h), str(w)), trimap)
            image = sqr_img
            bga = sqr_bga
            break

        count += 1  # debug usage

    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    bga = cv2.resize(bga, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

    return image, bga

# read/ crop/ resize inputs including fb, bg, alpha, trimap, composited image
# attention: all resize operation should be done before composition
def read_crop_resize(name, patch_size, stage):
    img_path, bg_path = name[0].strip(), name[1].strip()
    assert os.path.isfile(img_path) and os.path.isfile(bg_path), (img_path, bg_path)

    image = cv2.imread(img_path, -1)  # it's RGBA image
    fg = image[:,:,:3]
    a = image[:,:,3]
    bg = cv2.imread(bg_path)
    trimap = gen_trimap.rand_trimap(a, smooth=True)

    if stage == 't_net':
        image, bg = random_patch(image, trimap, patch_size, bg=bg)
        fg = image[:,:,:3]
        a  = image[:,:,3]
        # composite fg and bg, generate trimap
        img = composite(fg, a, bg)
        trimap = gen_trimap.rand_trimap(a, smooth=True)

        return img, trimap

    elif stage == 'm_net':
        fg, a, bg = mask_center_crop(fg, a, bg, trimap, patch_size)
        # generate trimap again to avoid alpha resize side-effect on trimap
        trimap = gen_trimap.rand_trimap(a, smooth=True)
        # composite fg and bg
        img = composite(fg, a, bg)

        return img, trimap, a, bg, fg

    elif stage == 'end2end':
        # for t_net
        t_patch = patch_size
        image_m, bg_m = random_patch(image, trimap, t_patch, bg=bg)
        fg_m = image_m[:,:,:3]
        a_m  = image_m[:,:,3]
        img_m = composite(fg_m, a_m, bg_m)
        trimap_m = gen_trimap.rand_trimap(a_m, smooth=True)
        '''
        # random flip and rotation before going to m_net
        bg_m = bg_m.astype(np.uint8)
        img_m, trimap_m, a_m, bg_m, fg_m = random_flip_rotation(img_m, trimap_m, a_m, bg_m, fg_m)

        # for m_net
        fg, a, bg = mask_center_crop(fg_m, a_m, bg_m, trimap_m, m_patch)
        # generate trimap again to avoid alpha resize side-effect on trimap
        trimap = gen_trimap.rand_trimap(a, smooth=True)
        # composite fg and bg
        img = composite(fg, a, bg)
        '''
        return img_m, trimap_m, a_m, bg_m, fg_m

# crop image into crop_size
def safe_crop(img, x, y, crop_size, resize_patch=None):
    if len(img.shape) == 2:
        new = np.zeros((crop_size, crop_size), np.uint8)
    else:
        new = np.zeros((crop_size, crop_size, 3), np.uint8)
    cropped = img[y:y+crop_size, x:x+crop_size]  # if y+crop_size bigger than len_y, return len_y
    h, w = cropped.shape[:2]  # h or w falls into the range of [crop_size/2, crop_size]
    new[0:h, 0:w] = cropped

    if resize_patch:
        new = cv2.resize(new, (resize_patch, resize_patch), interpolation=cv2.INTER_CUBIC)

    return new

# crop image around trimap unknown area with random size of [patch_size, 2*patch_size] then resize to patch_size
def mask_center_crop(fg, a, bg, trimap, patch_size):
    max_size = patch_size * 2
    min_size = patch_size
    crop_size = random.randint(min_size, max_size)  # both value are inclusive

    # get crop center around trimap unknown area
    y_idx, x_idx = np.where(trimap == 128)
    num_unknowns = len(y_idx)
    if num_unknowns > 0:
        idx = np.random.choice(range(num_unknowns))
        cx = x_idx[idx]
        cy = y_idx[idx]
        x = max(0, cx - int(crop_size / 2))
        y = max(0, cy - int(crop_size /2 ))
    else:
        raise ValueError('no unknown area in trimap!')

    # crop image/trimap/alpha/fg/bg
    fg = safe_crop(fg, x, y, crop_size, resize_patch=patch_size)
    a  = safe_crop(a, x, y, crop_size, resize_patch=patch_size)
    bg = safe_crop(bg, x, y, crop_size, resize_patch=patch_size)

    return fg, a, bg

# randomly flip and rotate images
def random_flip_rotation(image, trimap, alpha=None, bg=None, fg=None):
    if isinstance(alpha, np.ndarray) and isinstance(bg, np.ndarray):
        a_bg = True
    else:
        a_bg = False
    # horizontal flipping
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
        if a_bg:
            alpha = cv2.flip(alpha, 1)
            bg = cv2.flip(bg, 1)
            fg = cv2.flip(fg, 1)

    # rotation
    if random.random() < 0.05:
        degree = random.randint(1,3)  # rotate 1/2/3 * 90 degrees
        image = np.rot90(image, degree)
        trimap = np.rot90(trimap, degree)
        if a_bg:
            alpha = np.rot90(alpha, degree)
            bg = np.rot90(bg, degree)
            fg = np.rot90(fg, degree)

    if a_bg:
        return image, trimap, alpha, bg, fg
    else:
        return image, trimap
       
def np2Tensor(array):
    if len(array.shape)>2:
        ts = (2, 0, 1)
        tensor = torch.FloatTensor(array.transpose(ts).astype(float))
    else:
        tensor = torch.FloatTensor(array.astype(float))
    return tensor

class human_matting_data(data.Dataset):
    """
    human_matting
    """
    def __init__(self, args, split):
        super().__init__()
        self.patch_size = args.patch_size
        self.phase = args.train_phase

        if split == 'train':
            with open(args.train_fg_list, 'r') as f:
                self.path_fg=f.readlines()
        else:
            with open(args.val_fg_list, 'r') as f:
                self.path_fg=f.readlines()

        with open(args.bg_list, 'r') as f:
            self.path_bg = f.readlines()

        self.num = len(self.path_fg)
        self.bg_num = len(self.path_bg)

        print("Dataset : total {} images:{}".format(split, self.num))

    def __getitem__(self, index):
        # data structure returned :: dict {}
        # image: c, h, w / range[0-1] , float
        # trimap: 1, h, w / [0,1,2]  ,  float
        # alpha: 1, h, w / range[0-1] , float
        
        if self.phase == 'pre_train_t_net':
            # read files, random crop and resize
            image, trimap = read_crop_resize((self.path_fg[index],self.path_bg[random.randrange(0,self.bg_num)]), self.patch_size, stage='t_net')
            # augmentation
            image, trimap = random_flip_rotation(image, trimap)

            # NOTE ! ! ! trimap should be 3 classes for classification : fg, bg. unsure
            trimap[trimap == 0] = 0
            trimap[trimap == 128] = 1
            trimap[trimap == 255] = 2
            assert image.shape[:2] == trimap.shape[:2]

            # normalize
            image = image.astype(np.float32) / 255.0

            # to tensor
            image = np2Tensor(image)
            trimap = np2Tensor(trimap)

            trimap = trimap.unsqueeze_(0)  # shape: 1, h, w

            sample = {'image':image, 'trimap':trimap}

        elif self.phase == 'pre_train_m_net':
            # read files
            image, trimap, alpha, bg, fg = read_crop_resize((self.path_fg[index],self.path_bg[random.randrange(0,self.bg_num)]), self.patch_size, stage='m_net')
            # augmentation
            image, trimap, alpha, bg, fg = random_flip_rotation(image, trimap, alpha, bg, fg)

            # NOTE ! ! ! trimap should be 3 classes for classification : fg, bg. unsure
            trimap[trimap == 0] = 0
            trimap[trimap == 128] = 1
            trimap[trimap == 255] = 2

            assert image.shape[:2] == trimap.shape[:2] == alpha.shape[:2]

            # normalize
            image = image.astype(np.float32) / 255.0
            alpha = alpha.astype(np.float32) / 255.0
            bg = bg.astype(np.float32) / 255.0
            fg = fg.astype(np.float32) / 255.0

            # trimap one-hot encoding: when pre-train M_net, trimap should have 3 channels
            trimap = np.eye(3)[trimap.reshape(-1)].reshape(list(trimap.shape)+[3])

            # to tensor
            image = np2Tensor(image)
            trimap = np2Tensor(trimap)
            alpha = np2Tensor(alpha)
            bg = np2Tensor(bg)
            fg = np2Tensor(fg)

            alpha = alpha.unsqueeze_(0)   # shape: 1, h, w

            sample = {'image': image, 'trimap': trimap, 'alpha': alpha, 'bg':bg, 'fg':fg}

        elif self.phase == 'end_to_end':
            # read files
            assert len(self.patch_size) == 2, 'patch_size should have two values for end2end training !'
            img, trimap, a, bg, fg = read_crop_resize((self.path_fg[index],self.path_bg[random.randrange(0,self.bg_num)]), self.patch_size, stage='end2end')

            # NOTE ! ! ! trimap should be 3 classes for classification : fg, bg. unsure
            trimap[trimap == 0] = 0
            trimap[trimap == 128] = 1
            trimap[trimap == 255] = 2

            assert img.shape[:2] == trimap.shape[:2] == a.shape[:2]

            # t_net processing
            img = img.astype(np.float32) / 255.0
            img = np2Tensor(img)
            trimap = np2Tensor(trimap)
            trimap = trimap.unsqueeze_(0)  # shape: 1, h, w
            sample = {'image':img, 'trimap':trimap, 'alpha': a, 'bg':bg, 'fg':fg}

        return sample
    
    def __len__(self):
        return self.num

