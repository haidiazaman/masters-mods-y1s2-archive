import numpy as np
import scipy
from PIL import Image
import argparse
import matplotlib.pyplot as plt


def plot_figure(img_path, lvl, tap):
    n_rows,n_cols = 1,3
    fig,ax=plt.subplots(n_rows,n_cols,figsize=(10,20))

    # plot original image
    im = np.array(Image.open(img_path))
    plt.subplot(n_rows,n_cols,1)
    plt.imshow(im,cmap='gray')
    plt.title('original image')

    # plot intermediate coefficients
    coef = dwt2d(im, lvl, tap)
    plt.subplot(n_rows,n_cols,2)
    plt.imshow(coef,cmap='gray')
    plt.title('haar2d coefficients')

    # plot reconstructed image
    reconstructed_im = idwt2d(coef, lvl, tap)
    plt.subplot(n_rows,n_cols,3)
    plt.imshow(reconstructed_im,cmap='gray')
    plt.title('reconstructed image')
    
    plt.show()

def generate_filters(tap):
    
    # initialise h and g according to specific taps
    if tap==2:
        h = np.sqrt(1/2) * np.array([[1,1]]).T
        g = np.sqrt(1/2) * np.array([[1,-1]]).T
    elif tap==4:
        h = np.sqrt(1/2) * np.array([[0.6830127, 1.1830127, 0.3169873, -0.1830127]]).T
        g = np.sqrt(1/2) * np.array([[0.1830127, 0.3169873, -1.1830127, 0.6830127]]).T
    elif tap==6:
        h = np.sqrt(1/2) * np.array([[0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]]).T
        g = np.sqrt(1/2) * np.array([[-0.0498175, -0.12083221, 0.19093442, 0.650365, -1.14111692, 0.47046721]]).T
    else:
        print("the tap selected is not supported, select only 2, 4 or 6.")

    # generate the filters
    H = h @ h.T
    G1 = h @ g.T
    G2 = g @ h.T
    G3 = g @ g.T
    
    return H,G1,G2,G3


def dwt2d(im, lvl, tap):
    # Computing 2D discrete Haar wavelet transform of a given ndarray im.
    # Parameters: 
    #   im: ndarray.    An array representing image
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #   tap: integer.   An integer representing the tap of the wavelet filter
    #  Returns:
    #   out: ndarray.   An array representing  wavelet coefficients with lvl level. It has the same shape as im

    # generate filters based on tap specified
    H,G1,G2,G3 = generate_filters(tap)
    
    # reverse order of filter for the convolution operation
    H,G1,G2,G3 = H[::-1,::-1],G1[::-1,::-1],G2[::-1,::-1],G3[::-1,::-1]
    
    
    # pad zeros: an odd-length signal needs to be expanded to an even- length signal by zero-padding 
    if im.shape[0]%2==1:
        # check rows
        im = np.append(im,[[0]*im.shape[1]],axis=0)   
    if im.shape[1]%2==1:
        # check cols
        im = np.append(im,[[0]]*im.shape[0],axis=1)    
        
    
    # compute coef of the 2D discrete Haar wavelet
    out = np.zeros(shape=im.shape) # initialise a zero array to set coefficients
    s = im
    # convolution operation followed by downsampling for each filter
    for level in range(lvl):
        current_dim = int(len(s)/2)
        # compute w1
        out[current_dim:2*current_dim,:current_dim] = scipy.signal.convolve2d(s,G1,mode='same',boundary='wrap')[1::2,1::2]
        # compute w2
        out[:current_dim,current_dim:2*current_dim] = scipy.signal.convolve2d(s,G2,mode='same',boundary='wrap')[1::2,1::2]
        # compute w3
        out[current_dim:2*current_dim,current_dim:2*current_dim] = scipy.signal.convolve2d(s,G3,mode='same',boundary='wrap')[1::2,1::2]
        # compute s
        s = scipy.signal.convolve2d(s,H,mode='same',boundary='wrap')[1::2,1::2] 
        
    # set the first quadrant to be s at the end of the loop only
    out[:current_dim,:current_dim] = s 
    
    return out

def idwt2d(coef, lvl, tap):
    # Computing an image in the form of ndarray from the ndarray coef which represents its DWT coefficients.
    # Parameters: 
    #   coef: ndarray   An array representing 2D Haar wavelet coefficients
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #   tap: integer.   An integer representing the tap of the wavelet filter
    #  Returns:
    #   out: ndarray.   An array representing the image reconstructed from its  wavelet coefficients.

    
    # generate filters based on tap specified
    H,G1,G2,G3 = generate_filters(tap)
    
    # recover im from wavelet coef    
    out = np.zeros(shape=coef.shape) # initialise a zero array to reconstruct image   
    current_dim = int(coef.shape[0]/(2**lvl)) # find coef dim for highest level
    s = coef[:current_dim,:current_dim] # initialise s for this coef dim first
    
    for level in range(lvl):
        # get respective arrays of w1, w2, w3 - consistent with convention in dwt2d
        w1 = coef[current_dim:2*current_dim,:current_dim]
        w2 = coef[:current_dim,current_dim:2*current_dim]
        w3 = coef[current_dim:2*current_dim,current_dim:2*current_dim]
        
        # upsample 
        upsample_s = np.zeros(shape=(current_dim*2,current_dim*2))
        upsample_s[::2,::2] = s
        upsample_w1 = np.zeros(shape=(current_dim*2,current_dim*2))
        upsample_w1[::2,::2] = w1
        upsample_w2 = np.zeros(shape=(current_dim*2,current_dim*2))
        upsample_w2[::2,::2] = w2
        upsample_w3 = np.zeros(shape=(current_dim*2,current_dim*2))
        upsample_w3[::2,::2] = w3
        
        # convolution operation
        upsample_s = scipy.signal.convolve2d(upsample_s, H, mode='same', boundary='wrap')
        upsample_w1 = scipy.signal.convolve2d(upsample_w1, G1, mode='same', boundary='wrap')
        upsample_w2 = scipy.signal.convolve2d(upsample_w2, G2, mode='same', boundary='wrap')
        upsample_w3 = scipy.signal.convolve2d(upsample_w3, G3, mode='same', boundary='wrap')
        
        # sum coefficients
        s = upsample_s+upsample_w1+upsample_w2+upsample_w3        
        
        # multiply the current_dim by 2 to get dim for the following level
        current_dim *= 2 
        
    # set the out as s since s will be the reconstructed image after idwt2d through all levels
    # round the output to int values
    out = s.round().astype(np.uint8)

    return out


if __name__ == "__main__":
    # Code for testing.
    # Please modify the img_path to the path stored image and the level of wavelet decomposition.
    # Feel free to revise the codes for your convenience
    # If you have any question, please send email to e0983565@u.nus.edu for help
    # As the hw_1.pdf mentioned, you can also write the test codes on other .py file.

    parser = argparse.ArgumentParser(description="wavelet")
    parser.add_argument("--img_path",  type=str, default='./image_512.png',  help='The test image path')
    parser.add_argument("--level", type=int, default=4, help="The level of wavelet decomposition")
    parser.add_argument("--tap", type=int, default=2, help="The tap of the wavelet filter")
    parser.add_argument("--save_pth", type=str, default='./recovery.png', help="The save path of reconstructed image ")
    parser.add_argument("--plot_image", type=bool, default=False, help="Plot a 1x3 grid of images, in the order of original image, coefficients and reconstructed image for easy comparison.")
    opt = parser.parse_args()

    img_path = opt.img_path # The test image path
    level = opt.level # The level of wavelet decomposition
    tap = opt.tap # The tap of the wavelet filter
    save_pth = opt.save_pth
    plot_image = opt.plot_image

    img = np.array(Image.open(img_path).convert('L'))
    haar2d_coef = dwt2d(img,level, tap)
    recovery =  Image.fromarray(idwt2d(haar2d_coef, level, tap), mode='L')
    recovery.save(save_pth)
    np.save('./haar2_coeff.npy', haar2d_coef)
    
    if plot_image:
        plot_figure(img_path, level, tap)