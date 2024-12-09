import numpy as np
from PIL import Image
import cv2



def _get_all_pairs_train(file_path: str):
    
    pairs_image = np.load(file_path, allow_pickle=True)
    return pairs_image

def get_pairs_train(file_path: str, dir: str):
    """
    Args:
        file_path: path to file in the form train_pairs_noflip.npy
        dir: directory name containing the objects to handle doppelgangers, eg: Arc_de_Triomphe_du_Carrousel_by_angle

    Return:
        A list of pairs of paths to images of the specified object, along with the sequence number in the original train set
    """
    pairs_all_image = _get_all_pairs_train(file_path)

    pairs_img = []

    for i, pair_image in enumerate(pairs_all_image):
        img1_path, img2_path = pair_image[:2]

        if img1_path.split('/')[0] == dir:

            pairs_img.append((pair_image, i))

    return pairs_img

def get_keypoints_location_pair(parent_path, indice: int):
    """
    Args:
        parent_path: path to the file containing the .npy file that needs to identify keypoints location pair
        indice: pair number in the train set

    Return:
        A dict containing kpt1, kpt2, conf
    """
    file_path = parent_path.strip('/') + '/' + str(indice) + '.npy'

    loftr = np.load(file_path, allow_pickle=True)

    return loftr.tolist()

def get_images(images_path, root_path_images=''):
    images = []
    for image_path in images_path:
        images.append(
            np.array(Image.open(root_path_images.strip('/') + '/' + image_path))
        )
    return images

def resize_and_pad_bottom_right(image, target_size=(1024, 1024), pad_color=(0, 0, 0)):
    """
    Resize and pad an image to the target size with padding on the bottom and right.

    Args:
        image: Input image as a numpy array.
        target_size: Tuple (width, height) for the output size.
        pad_color: Padding color (B, G, R).

    Returns:
        result: Resized and padded image.
    """
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale and resize
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding (only for bottom and right)
    pad_bottom = target_h - new_h
    pad_right = target_w - new_w

    # Apply padding
    result = cv2.copyMakeBorder(
        resized_image,
        top=0, bottom=pad_bottom,
        left=0, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return result

def crop_matched_regions(image1, image2, kpt0, kpt1, conf, threshold=0.5):
    """
    Crop the matching area between 2 images based on kpt0, kpt1 and confidence (conf).

    Args:
        image1: Input image 1 (numpy array).
        image2: Input image 2 (numpy array).
        kpt0: Feature points in image 1 (numpy array, shape: [N, 2]).
        kpt1: Feature points in image 2 (numpy array, shape: [N, 2]).
        conf: Confidence score of feature point pairs (numpy array, shape: [N]).
        threshold: Confidence score threshold to filter points (default: 0.5).

    Returns:
        cropped1: Cropped area in image 1.
        cropped2: Cropped area in image 2.
    """
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    # Filter feature points by confidence
    valid_idx = conf >= threshold
    kpt0_valid = kpt0[valid_idx]
    kpt1_valid = kpt1[valid_idx]
    

    # Calculate the bounding box of feature points
    x_min0, y_min0 = kpt0_valid.min(axis=0).astype(int)
    x_max0, y_max0 = kpt0_valid.max(axis=0).astype(int)

    x_min1, y_min1 = kpt1_valid.min(axis=0).astype(int)
    x_max1, y_max1 = kpt1_valid.max(axis=0).astype(int)
    # print((x_min1, y_min1), (x_max1, y_max1))

    # Crop the area in image 1 and image 2
    cropped1 = image1[y_min0:y_max0, x_min0:x_max0]
    cropped2 = image2[y_min1:y_max1, x_min1:x_max1]

    return cropped1, cropped2

def write_keypoints_to_image(image, kpts):
    for x, y in kpts:
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

    return image