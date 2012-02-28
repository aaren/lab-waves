import Image
import ImageDraw
import os

def sanity_check(inter, maxima, minima, front, image):
    """produces an image on screen that has the calculated interface
    and the inferred peak coordinates and front position overlaid.
    
    args: inter -- the interface depth list
          maxima -- the list of maxima coordinates
          minima -- the list of minima coordinates
          front -- the coordinate of the current front

    return: none. opens a pil image object in xv and imagemagick
    """
    # (half) the size of the rectangles to plot at the max and min locations
    rectangle_size = (7,7)

    x_coords = range(len(inter))
    y_coords = inter

    i_coords = zip(x_coords, y_coords)

    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    # plot the measured interface depth onto the image
    draw.line(i_coords, fill = 'black', width = 5)
 
    # plot red and green squares onto the image at the calculated 
    # maxima and minima
    for coord in maxima:
        xy = [(coord[0] + rectangle_size[0], coord[1] + rectangle_size[1]), \
                (coord[0] - rectangle_size[0], coord[1] - rectangle_size[1])]
        draw.rectangle(xy, fill = 'red')
    for coord in minima:
        xy = [(coord[0] + rectangle_size[0], coord[1] + rectangle_size[1]), \
                (coord[0] - rectangle_size[0], coord[1] - rectangle_size[1])]
        draw.rectangle(xy, fill = 'green')
    # put a blue square where the front position is
    for coord in front:
        xy = [(coord[0] + rectangle_size[0], coord[1] + rectangle_size[1]), \
                (coord[0] - rectangle_size[0], coord[1] - rectangle_size[1])]
        draw.rectangle(xy, fill = 'blue')

    # for some reason this produces output in both xv
    # and imagemagick concurrently
    # im.show()
    # command to kill the imagemagick instance
    # killit = 'killall display'
    # os.system(killit)

    run = image.split('/')[0]
    camera = image.split('/')[1]
    frame = image.split('/')[2]
    
    sanity_dir = run + '/' + camera + '_sanity/'
    if not os.path.exists(sanity_dir):
        os.makedirs(sanity_dir)
    else:
        pass


    im.save(sanity_dir + frame)
    print "wrote out sanity image to", sanity_dir, frame

