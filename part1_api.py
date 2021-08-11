import scipy

try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise

"""
This function get a gray image and return vectors of specific point
suspected as bright point 
"""


def find_lights(gray_color):
    # convolve gray image with a kernel that presentation a circle
    circle_kernel = np.array([[2 / 3, -2 / 3, -2 / 3], [1 / 2, -1, 1 / 2], [2 / 3, 2 / 3, -2 / 3]])
    circle_kernel.astype('float64')
    convolve_res = sg.convolve(gray_color, circle_kernel, mode="same")
    # Get maximum filter in 100 range
    maximum_filter_res = scipy.ndimage.maximum_filter(convolve_res, size=100)
    # Subtract between maximum_filter_res and convolve_res to get the specific point
    subtract_res = np.subtract(maximum_filter_res, convolve_res)
    # Get a coordinates of point that greater than 220 in convolve_res and equal to 0 in subtract_res
    x, y = np.where((subtract_res ==0) & (convolve_res > 150))
    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    c_image.astype('float64')
    # Extraction of the red layer and send it to find_lights function
    gray_color_red = c_image[:, :, 0]
    x_red, y_red = find_lights(gray_color_red)
    # Extraction of the green layer and send it to find_lights function
    gray_color_green = c_image[:, :, 1]
    x_green, y_green = find_lights(gray_color_green)
    return y_red, x_red, y_green, x_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    # default_base = './data/leftImg8bit/test/berlin'
    default_base = './data/leftImg8bit/test/berlin'

    if args.dir is None:
        args.dir = default_base
    f_list = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in f_list:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(f_list):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
