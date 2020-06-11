import math
import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def paa_segmentation(data, image_width, overlapping=False, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.
    """
    if data.ndim==1:
        data = data[:,None]
    n_samples, n_timestamps = data.shape
    window, remainder = divmod(n_samples, image_width)
    window_size = int(np.floor(n_samples//image_width))
    #window_size = window if remainder == 0 else window + 1

    if n_segments is None:
        quotient, remainder = divmod(n_samples, window_size)
        n_segments = quotient if remainder == 0 else quotient - 1
        
    if not overlapping:
        bounds = np.linspace(0, n_samples, n_segments + 1).astype('int64')
        start = bounds[:-1]
        end = bounds[1:]
        size = start.size
    else:
        n_overlapping = (n_segments * window_size) - n_samples
        n_overlaps = n_segments - 1
        overlaps = np.linspace(0, n_overlapping,
                               n_overlaps + 1).astype('int64')
        bounds = np.arange(0, (n_segments + 1) * window_size, window_size)
        start = bounds[:-1] - overlaps
        end = bounds[1:] - overlaps
        size = start.size
    
    return np.array([np.median(data[start[i]:end[i]]) for i in range(size)])


def get_rec_plot(s, eps=0.10, steps=10, distance='euclidean', binary=False):
    
    if s.ndim==1:
        s = s[:,None]
   
    d = pdist(s, metric=distance)
    if steps<1:
        d[d>=eps]=1
        d[d<eps]=0
    else:
        d = np.floor(d/eps)
        d[d>steps] = steps
    Z = squareform(d)
    return Z

def generateRPImage(c, w, eps=0.1, steps=10, distance='euclidean', binary=False):
    i_aa=paa_segmentation(c, w, overlapping=True, n_segments=None)
    img = get_rec_plot(i_aa, eps, steps, distance, binary)
    #img = img/np.max(img)
    img = rescale_image(img)
    return img

def get_binary_vi(c,v, bins=50):
    xi = 0
    xbins = np.linspace(-1, 1, num=bins+1)
    ybins = np.linspace(-1, 1, num = bins+1)
    m = np.zeros((bins,bins))
    for x1, x2 in zip(xbins[:-1],xbins[1:]):
        yi = bins-1
        for y1, y2 in zip(ybins[:-1],ybins[1:]):
            m[yi, xi] = sum((x1 <= c) & (c < x2) &
                                       (y1 <= v) & (v < y2))
            yi -= 1
        xi += 1
    m = m / np.max(m)
    
    return m

def generateBinaryimage(c, v, w=16, para=0.5, threshold=0, rescale=1):
    
    """
    Generate I-V binary image
    Agg:import argparse
       cimport argparse
       vimport argparse
       wimport argparse
       pimport argparse
       timport argparse
       rescale:bool wether to rescale image
    Return:
       Image
    """

    #find min and max voltage
    v_min=np.min(v)
    v_max=np.max(v)
    
    #find min and max current
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #c=scaler.fit_transform(c.reshape(-1,1))
    c_min=np.min(c)
    c_max=np.max(c)

    #get max value of current and voltage
    d_c = max(abs(c_min),c_max)
    d_v = max(abs(v_min),v_max)

    scaling_factor=d_c

    #Resize current value to d_c value
    c[c<-d_c] = -d_c
    c[c>d_c] = d_c
    
    #Resize voltage value to d_c value
    v[v<-d_v] = -d_v
    v[v>d_v] = d_v

    d_c = (c_max-c_min)/(w-1)
    d_v = (v_max-v_min)/(w-1)


    #ind_c = np.ceil((c-np.amin(c))/d_c)
    
    ind_c = np.ceil((c-np.min(c))/d_c) if d_c>0 else np.ceil((c-np.min(c)))
    ind_v = np.ceil((v-np.min(v))/d_v) if d_v>0 else np.ceil((v-np.min(v)))
    ind_c[ind_c==w] = w-1
    ind_v[ind_v==w] = w-1
    
    #create image data
    Img = np.zeros((w,w))  
    for i in range(len(c)):
        Img[int(ind_c[i]),int(w-ind_v[i]-1)] += 1

    if rescale:
        Img = (float(w-1) / Img.max() * (Img -Img.min()))
    if threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
   
    Img = rescale_image(Img)
    return Img



def center(X,w):
    minX = np.amin(X)
    maxX = np.amax(X)
    dist = max(abs(minX),maxX)
    X[X<-dist] = -dist
    X[X>dist] = dist
    d = (maxX-minX)//(w-1)
    return (X,d)
    
def get_img_from_VI(V, I, width,hard_threshold=True,para=.5):
    '''Get images from VI, hard_threshold, set para as threshold to cut off,5-10
    soft_threshold, set para to .1-.5 to shrink the intensity'''
    # center the current and voltage, get the size resolution of mesh given width
    d = V.shape[0]
    # doing interploation if number of points is less than width*2
    if d<2* width:
        newI = np.hstack([V, V[0]])
        newV = np.hstack([I, I[0]])
        oldt = np.linspace(0,d,d+1)
        newt = np.linspace(0,d,2*width)
        I = np.interp(newt,oldt,newI)
        V = np.interp(newt,oldt,newV)
        
    (I,d_c)  = center(I,width)
    (V,d_v)  = center(V,width)
    
    #  find the index where the VI goes through in current-voltage axis
    ind_c = np.ceil((I-np.amin(I))/d_c)
    ind_v = np.ceil((V-np.amin(V))/d_v)
    ind_c[ind_c==width] = width-1
    ind_v[ind_v==width] = width-1
    
    Img = np.zeros((width,width))
    
    for i in range(len(I)):
        Img[int(ind_c[i]),int(width-ind_v[i]-1)] += 1
    
    if hard_threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
        
    else:
        Img=Img/(np.max(Img)**para)
    return rescale_image(Img)
 
def createImage(current, voltage, width=50,image_type="vi", eps=1e-1, steps=10, distance='euclidean', binary=False):
    
    n = len(current)
    Imgs = np.empty((n,width,width), dtype=np.float64)
    for i in range(n):
        if image_type=="vi":
            Imgs[i,:,:] = generateBinaryimage(current[i],voltage[i,], width,True,1)
        else:
            Imgs[i,:,:] = generateRPImage(current[i,],  width, eps, steps, distance, binary)
    
    return np.reshape(Imgs,(n,width, width,1))


def createRPImage(current, voltage, width=50, eps=1e-1, steps=10, distance='euclidean', binary=False):
    
    n = len(current)
    Imgs = np.empty((n,width,width, 2), dtype=np.float64)
    for i in range(n):
       
        c_im= generateRPImage(current[i,],  width, eps, steps, distance, binary)
        v_im= generateRPImage(voltage[i,],  width, eps, steps, distance, binary)
        img = np.concatenate([c_im[:, :, np.newaxis],v_im[:, :, np.newaxis]], 2)
        Imgs[i,:,:] = img
    
    return Imgs

def createVImage(current, voltage, width=50):
    
    n = len(current)
    Imgs = np.empty((n,width,width), dtype=np.float64)
    progress_bar = tqdm(np.arange(n))
    for i in progress_bar:
        progress_bar.set_description('ID ' + str(i+1))
        Imgs[i,:,:] = get_binary_vi(current[i,], voltage[i,], bins=width)
        #Imgs[i,:,:] = get_img_from_VI(voltage[i,], current[i,], width,hard_threshold=True,para=.5)
        #progress_bar.set_postfix('processed: %d' % (1 + i))
    
    return np.reshape(Imgs,(n,width, width,1))




def rescale_image(img, range=(0, 255)):
    scaler = MinMaxScaler(feature_range=range)
    return scaler.fit_transform(img).astype(np.uint8)


def calculatePower(train_current, train_voltage,  NN):
   
    
    seq_len = int(NN)
    total_size = len(train_current)-seq_len
    max_seq = total_size//seq_len
    active_power = []
    apprent_power = []
    PQ = np.empty([max_seq,2])
   
    PQ
    for idx in range(0, total_size, seq_len):

        size = slice(idx, idx + seq_len)
        temp_I = train_current[size]
        temp_V = train_voltage[size]
    
        
        Irms = np.mean(temp_I**2)**0.5
        Vrms = np.mean(temp_V**2)**0.5
        P = np.mean(temp_I * temp_V)
        S  = Vrms*Irms
        
        
        active_power+=[P]
        apprent_power+=[S]
        
    
    PQ = np.array([active_power, apprent_power])
     

    return PQ.T

