import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_edgemap(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image and reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply the opening operation to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(opening, 50, 150)
    
    return edges


def get_approximate_corners_from_contours(img,edges):
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract the corner points of the polygon
    corners = np.squeeze(approx)
        
    return corners


def plot_contours(img,corners):
    # create deepcopy of the img
    img2 = img[:,:,:]

    # Draw the contour on the original image
    cv2.drawContours(img2, [corners.reshape(4,1,2)], -1, (0, 255, 0), 10)

    # Draw the corner points on the original image
    for corner in corners:
        cv2.circle(img2, tuple(corner), 20, (255, 0, 0), -1)

    # Display the result
    plt.imshow(img2)
    plt.show()


def get_warped_image(img,corners):
    
    # Define object height and width
    top_r,top_l,bot_l,bot_r = corners # same format across different images
    object_height = min(bot_l[1]-top_l[1],bot_r[1]-top_r[1])
    object_width = min(bot_r[0]-bot_l[0],top_r[0]-top_l[0])
    # because of the perspective of different images, the edge nearer to camera will appear bigger
    # as such, should select the smaller of the 2 parallel edges
    
    # Define the output rectangular points 
    output_height = img.shape[0]  
    output_width = img.shape[1] 
    img_centre_x,img_centre_y = output_width//2,output_height//2
    
    # Assign the coordinates of the target corners accordingly
    target_top_r = [img_centre_x+object_width//2,img_centre_y-object_height//2]
    target_top_l = [img_centre_x-object_width//2,img_centre_y-object_height//2]
    target_bot_r = [img_centre_x+object_width//2,img_centre_y+object_height//2]
    target_bot_l = [img_centre_x-object_width//2,img_centre_y+object_height//2]
    target = np.array([target_top_r,target_top_l,target_bot_l,target_bot_r], dtype=np.float32) 

    # Estimate the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)

    # Warp the image using the perspective transform
    warped_image = cv2.warpPerspective(img, perspective_matrix, (output_width, output_height), borderMode=cv2.BORDER_REPLICATE)

    return warped_image


def imrect(im1):
    # Perform Image rectification on an 3D array im.
    # Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
    # Returns: out: numpy.ndarray, rectified image。
    
    # Get edgemap using Canny edge detector
    edges = get_edgemap(im1)
    
    # Get approximate corners of the object of interest from its contours
    corners = get_approximate_corners_from_contours(im1,edges)
    
    # Get the warped image from the img and corners
    out = get_warped_image(im1,corners)

    return out


if __name__ == "__main__":
    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = cv2.imread(name)
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:].replace('.jpg', '.png'),np.uint8(np.clip(np.around(rectificated,decimals=0),0,255)))

        # # UNCOMMENT THIS BLOCK - to see the detected contours and corners of the original image
        # # Get edgemap using Canny edge detector
        # edges = get_edgemap(image)
        # # Get approximate corners of the object of interest from its contours
        # corners = get_approximate_corners_from_contours(image,edges)
        # plot_contours(image,corners)