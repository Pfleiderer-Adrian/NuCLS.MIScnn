# ==============================================================================#
#  Author:       Adrian Pfleiderer                                             #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# ==============================================================================#


import matplotlib.pyplot as plt
import os

from PIL import Image
import numpy as np
from skimage import io, color

def drawContour(m, s, c, RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p: p == c and 255)

    # Find edges of this contour and make into Numpy array
    # thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN = np.array(thisContour)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m


def overlay_seg(img_path, seg_path, pred_path, seg_code, seg_name, save_path):
    # Load segmented image as greyscale
    seg = Image.open(seg_path).convert('L')
    pred = Image.open(pred_path).convert('L')

    # Load main image - desaturate and revert to RGB so we can draw on it in colour
    main = Image.open(img_path).convert('L').convert('RGB')

    mainN = np.array(main)
    mainN2 = np.array(main)

    mainN = drawContour(mainN, seg, seg_code, (255, 0, 0))  # draw contour 1 in red
    mainN2 = drawContour(mainN2, pred, seg_code, (255, 0, 0))  # draw contour 1 in red

    # Save result
    f, axarr = plt.subplots(1, 3)
    f.suptitle("Klasse: "+seg_name, fontsize=14)
    axarr[0].imshow(main)
    axarr[0].set_title("Original")
    axarr[1].imshow(Image.fromarray(mainN))
    axarr[1].set_title("Ground Truth")
    axarr[2].imshow(Image.fromarray(mainN2))
    axarr[2].set_title("Prediction")

    plt.savefig(os.path.join(save_path, seg_name + '.pdf'))
    plt.close(f)


def create_superclass_ground_truth(mainN, segN, save_path, save_name):
    segN = segN[:,:,0]
    colors = [(238,180,180), (151,205,205), (139,0,0), (139,0,139), (255,185,15)]
    plt.imshow(color.label2rgb(segN, mainN, alpha=0.5, colors=[(255,0,0),(0,0,255), (0, 255, 0), (255,0 ,255), (255,255,0)], bg_label=0, bg_color=None))
    plt.savefig(os.path.join(save_path,  save_name+'.pdf'))
    plt.clf()

def save_image(plot, path, name):
    plot.imsave(path + name + ".png")
