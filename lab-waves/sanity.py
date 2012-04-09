import os
import sys

import Image
import ImageDraw

def sanity_check(interfaces, points, image, icolours, pcolours):
    """produces an image that has the calculated interface
    and the inferred peak coordinates and front position overlaid.
    
    args: interfaces -- a list of interface depth lists
          maxima -- the list of maxima coordinates
          minima -- the list of minima coordinates
          front -- the coordinate of the current front
          colous -- a list of colours with which to draw the interfaces

    return: none. Saves an image in the sanity_dir
    """

    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    # plot the measured interface depth onto the image
    for inter, colour in zip(interfaces, icolours):
        x_coords = range(len(inter))
        y_coords = inter
        i_coords = zip(x_coords, y_coords)
        draw.line(i_coords, fill = colour, width = 5)
 
    # plot squares onto the image at the given points
    rectangle_size = (7,7)
    for point, colour in zip(points, pcolours):
        for coord in point:
            xy = [(coord[0] + rectangle_size[0], coord[1] + rectangle_size[1]), \
                    (coord[0] - rectangle_size[0], coord[1] - rectangle_size[1])]
            draw.rectangle(xy, fill = colour)

    run = image.split('/')[-3]
    camera = image.split('/')[-2]
    frame = image.split('/')[-1]
    
    # derive the root data directory from the image filename that is passed
    root_data_dir = ('/').join(image.split('/')[:-3]) + '/'
    sanity_dir = root_data_dir + run + '/' + camera + '_sanity/'
    if not os.path.exists(sanity_dir):
        os.makedirs(sanity_dir)
    else:
        pass

    im.save(sanity_dir + frame)
    print "wrote sanity image to",sanity_dir,frame,"\r",
    sys.stdout.flush()
