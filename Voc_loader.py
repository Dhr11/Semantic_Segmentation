import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import imageio
import scipy.misc as m

from PIL import Image
from tqdm import tqdm
from torch.utils import data


def bytescale(data, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255 or low < 0 or high < low:
        raise ValueError("check high low values")
    
    cmin = data.min()
    cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        #mode in [None, 'L', 'P']:
        bytedata = bytescale(data, high=high, low=low)
        image = Image.frombytes('L', shape, bytedata.tostring())
        return image
        
    
class VOCLoader(data.Dataset):
    files = {}
    labels = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ])
    def __init__(self,root,img_size=[384,384],portion="train",n_class = 21,do_transform=False,do_norm=True,test=False):
        self.root = root
        self.img_size = img_size
        self.portion = portion
        #self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.n_class = n_class
        self.do_transform = do_transform
        self.do_norm = do_norm
        self.test = test
        self.setup(root)
    def __len__(self):
        return len(self.files[self.portion])    
    """
    def getvocpallete(self,num_cls):
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete
    """
    def setup(self,root):
        self.get_filenames(root)
        if not os.path.exists(os.path.join(root,"SegmentationClass/encoded/")):
            os.makedirs(os.path.join(root,"SegmentationClass/encoded/"))
        else:
            return    
        for dataset in self.files:
            for filename in tqdm(self.files[dataset]):
                filepath = os.path.join(root,"SegmentationClass/",filename+".png")
                label_mask = self.encode(imageio.imread(filepath,as_gray=False,pilmode="RGB"))
                #label_mask = (label_mask * 255).astype(np.uint8)
                lbl = toimage(label_mask,high=label_mask.max(),low=label_mask.min())#,mode="L")#, high=label_mask.max(), low=label_mask.min())
                #lbl.putpalette(self.getvocpallete(256))
                imageio.imwrite(os.path.join(root,"SegmentationClass/encoded/",filename+".png"), lbl)
    def get_filenames(self,root):
        
        for dataset in ["train","val"]:
            filename = os.path.join(root,"ImageSets/Segmentation/",dataset+".txt")
            fo = open(filename, "r+")
            names = [i.strip() for i in fo.readlines()]
            self.files[dataset] = names

    def encode(self, mask):
    ## mask is in RGB R,C,3
        #print(mask.shape)
        mask = mask.astype(int)  
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.labels):
            #print(ii,label.shape,np.where(np.all(mask == label, axis=-1))[:2])
            
            ## Checks along one 2d pixel location, for colour r,g,b values and matches with class colours
            locations = np.all(mask == label, axis=-1)
            label_mask[locations]=ii 
        return label_mask.astype(int)

    def decode(self,label_mask_gen):
        label_mask_gen = np.expand_dims(label_mask_gen,axis= -1)
        img_3d = np.concatenate((label_mask_gen,label_mask_gen,label_mask_gen),axis = -1)
        for ii, label in enumerate(self.labels):

            locations = np.all(label_mask_gen == ii,axis=-1)
            #print(locations,locations.shape)
            img_3d[locations]=label
        #img_3d = np.divide(img_3d,255.0)#img_3d/255.0
        return img_3d
    def __getitem__(self,index):
        filename = self.files[self.portion][index]
        org_img = os.path.join(self.root,"JPEGImages",filename+".jpg")
        label_mask = os.path.join(self.root,"SegmentationClass/encoded/",filename+".png")
        org_img = Image.open(org_img)
        label_mask = Image.open(label_mask)
        if self.do_transform:
            transform = transforms.Compose(
            [   transforms.Resize([256,256]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            org_img = transform(org_img)
            label_mask = transforms.Resize([256,256])(label_mask)
            label_mask = torch.from_numpy(np.array(label_mask)).long()
            label_mask[label_mask == 255] = 0
        #org_img = org_img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        #label_mask = label_mask.resize((self.img_size[0], self.img_size[1]))
        return org_img, label_mask    
#encode(np.array(imageio.imread("G:/VOC_Segmentation/VOCdevkit/VOC2012/SegmentationClass/2007_000042.png",as_gray=False,pilmode="RGB")))
#res = decode(np.array([[1,2],[3,4],[5,6]]))
#res[:,:,2]



"""
Debuggging functions

voc = VOCLoader("./")
res = voc.decode(np.array([[1,2],[3,4],[5,6]]))
res
voc.get_filenames("./")
voc.files["train"]

sample_img = imageio.imread("G:/VOC_Segmentation/VOCdevkit/VOC2012/SegmentationClass/2007_000042.png",as_gray=False,pilmode="RGB")
label_mask = voc.encode(sample_img)

lbl = toimage(label_mask,high=label_mask.max(),low=label_mask.min())#,mode="L")#, high=label_mask.max(), low=label_mask.min())
print(label_mask[100,192],sample_img[100,192,:])

imageio.imwrite(os.path.join("./","SegmentationClass/encoded/","2007_000042_test"+".png"), lbl)
load_img = imageio.imread("G:/VOC_Segmentation/VOCdevkit/VOC2012/SegmentationClass/encoded/2007_000042_test.png",as_gray=False)
print(load_img[100,192],np.sum(lbl),np.sum(load_img))

"""