title: correcting barrel distortion in python
date: 24/12/12

Barrel distortion can be corrected in python with the following:

    :::python
    def r_from_centre((x, y), (w, h)):
        """Calculate the radius from the centre of an image of a point (x, y)
        in the image.
        """
        r = ((w / 2 - x) ** 2 + (h / 2 - y) ** 2) ** .5
        return r

    def barrel_dist_r((x, y), (w, h), (a, b, c, d)):
        """For given point x, y in the desination, calculate the radius
        to the equivalent point in the source.

        The radius is normalised such that the described circle will fit
        entirely into the image.

        i.e. half of the smallest of h, w is used to normalise.
        """
        r_dest = r_from_centre((x, y), (w, h))
        norm = min(w,h) / 2
        norm_r_dest = r_dest / norm
        norm_r_src = barrel_dist(norm_r_dest, (a, b, c, d))
        r_src = norm_r_src * norm
        return r_src

    def barrel_dist(r, (a, b, c, d)):
        """For given radius in the destination, calculate the equivalent
        radius in the source for given coefficients.

        r has to be in units normalised to half the minimum side length.

        That is, the circle that r describes must fit entirely into the image.
        """
        r_src = r * (a * r ** 3 + b * r ** 2 + c * r + d)
        return r_src

    def f((x,y,z), (w, h), (a,b,c,d)):
        """Map destination pixels to source pixels for barrel
        distortion coefficients a, b, c, d and an image of
        dimensions (w, h).

        There are two equations to solve.

        Given an (x, y) input we can calculate the radius of this point
        in both the source and desination images.

        Secondly, we conserve the direction of the vector from the centre
        of the image to the point in both images.

        i.e. (w / 2 - x_src) / (h / 2 - y_src) = (w / 2 - x_dest) / (h / 2 - y_dest)

        This gives us a quadratic in y_src.
        """
        print '\r', x, y,
        if x == w / 2 and y == h / 2:
            return x, y, z
        elif y == h / 2:
            x_src = barrel_dist_r((x, y), (w, h), (a, b, c, d))
            y_src = y
            return x_src, y_src, z

        tan = (w / 2 - x) / (h / 2 - y)

        r = barrel_dist_r((x, y), (w, h), (a, b, c, d))
        # coefficients A x ** 2 + B * x + C = 0
        A = 1 + tan ** 2
        B = (h * (1 + tan ** 2) - 2 * w * tan)
        C = w ** 2 / 2 + (h ** 2 / 4) * (1 + tan) + (w * h / 2) * tan - r ** 2

        roots = np.roots((A, B, C))
        # this is the correct root selection. It comes out like this
        # because of the simple formula used to equalise the angles.
        if x < w / 2 and y < h / 2:
            y_src = roots[1]
        elif x < w / 2 and y > h / 2:
            y_src = roots[0]
        elif x >= w / 2 and y < h / 2:
            y_src = roots[0]
        elif x >= w / 2 and y >= h / 2:
            y_src = roots[0]
        else:
            exit("I don't know what you're talking about")

        x_src = tan * (y_src - h / 2) + w / 2

        return x_src, y_src, z

    corr = (0.000658776, -0.0150048, -0.00123339, 1.01557914)
    im = ndimage.imread(image)
    dim = im.shape[0:2]
    imcorr = geometric_transform(im, f, extra_arguments=(dim, corr))


However, this is really slow. `timeit` gives 278us for the function `f`.

    Timer unit: 1e-06 s

    File: proc_im.py
    Function: f at line 172
    Total time: 0.001848 s

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
    172         1           14     14.0      0.8  def f((x,y,z), (w, h), (a,b,c,d)):
    173                                               """Map destination pixels to source pixels for barrel
    174                                               distortion coefficients a, b, c, d and an image of
    175                                               dimensions (w, h).
    176                                           
    177                                               There are two equations to solve.
    178                                           
    179                                               Given an (x, y) input we can calculate the radius of this point
    180                                               in both the source and desination images.
    181                                           
    182                                               Secondly, we conserve the direction of the vector from the centre
    183                                               of the image to the point in both images.
    184                                           
    185                                               i.e. (w / 2 - x_src) / (h / 2 - y_src) = (w / 2 - x_dest) / (h / 2 - y_dest)
    186                                           
    187                                               This gives us a quadratic in y_src.
    188                                               """
    189         1           33     33.0      1.8      print '\r', x, y,
    190         1           11     11.0      0.6      if x == w / 2 and y == h / 2:
    191                                                   return x, y, z
    192         1            9      9.0      0.5      elif y == h / 2:
    193                                                   x_src = barrel_dist_r((x, y), (w, h), (a, b, c, d))
    194                                                   y_src = y
    195                                                   return x_src, y_src, z
    196                                           
    197         1           12     12.0      0.6      tan = (w / 2 - x) / (h / 2 - y)
    198                                           
    199         1           98     98.0      5.3      r = barrel_dist_r((x, y), (w, h), (a, b, c, d))
    200                                               # coefficients A x ** 2 + B * x + C = 0
    201         1           12     12.0      0.6      A = 1 + tan ** 2
    202         1           12     12.0      0.6      B = (h * (1 + tan ** 2) - 2 * w * tan)
    203         1           15     15.0      0.8      C = w ** 2 / 2 + (h ** 2 / 4) * (1 + tan) + (w * h / 2) * tan - r ** 2
    204                                           
    205         1         1559   1559.0     84.4      roots = np.roots((A, B, C))
    206                                               # this is the correct root selection. It comes out like this
    207                                               # because of the simple formula used to equalise the angles.
    208         1           13     13.0      0.7      if x < w / 2 and y < h / 2:
    209                                                   y_src = roots[1]
    210         1           10     10.0      0.5      elif x < w / 2 and y > h / 2:
    211         1           11     11.0      0.6          y_src = roots[0]
    212                                               elif x >= w / 2 and y < h / 2:
    213                                                   y_src = roots[0]
    214                                               elif x >= w / 2 and y >= h / 2:
    215                                                   y_src = roots[0]
    216                                               else:
    217                                                   exit("I don't know what you're talking about")
    218                                           
    219         1           30     30.0      1.6      x_src = tan * (y_src - h / 2) + w / 2
    220                                           
    221         1            9      9.0      0.5      return x_src, y_src, z

We can see that the majority of the time is spent in `np.roots`, so let's
calculate those explicitly as it's only a quadratic,

    205         1           14     14.0      3.6      r1 = (-B + (B ** 2 - 4 * A * C) ** .5) / (2 * A)
    206         1           13     13.0      3.3      r2 = (-B - (B ** 2 - 4 * A * C) ** .5) / (2 * A)
    207         1            8      8.0      2.0      roots = (r1, r2)

Now `timeit` is 24us - an order of magnitude faster.

In fact, if we're a bit more clever with the equations we can get
this down to 13us because we don't actually have to form a polynomial
at all:

    :::python
    def f((x,y), (w, h), (a,b,c,d)):
        if x == w / 2 and y == h / 2:
            return x, y, z
        r_dest = ((w / 2 - x) ** 2 + (h / 2 - y) ** 2) ** .5
        nr = r_dest / (min(w, h) / 2)  # normalise
        # ratio r is nr_src / nr_dest
        r = (a * nr ** 3 + b * nr ** 2 + c * nr + d)

        x_src = r * x + (w / 2) * (1 - r)
        y_src = r * y + (w / 2) * (1 - r)

        return x_src, y_src

There aren't any more big efficiencies to be made. We could make the
`barrel_dist_r` function call explicit, but that isn't going to give
us another order of magnitude improvement.

The mapping is being applied with scipy's `geometric_transform`, which
works over all of the pixels in the image. Instead of mapping each pixel
we could divide the image into a coarse grid and just map blocks of that.

This has been [asked][] before:

[asked]: http://mail.python.org/pipermail/image-sig/2010-May/006254.html

On Sat, May 8, 2010 at 5:58 AM, Son Hua <songuke at gmail.com> wrote:
> Hi,
>
> Can anybody show me some hints on how to implement an image warping
> function
> efficiently in Python or PIL? Suppose we have a function f(x, y) ->
> (x', y')
> that warps a pixel location (x, y) to (x', y'). Because (x', y') may
> end up
> to be non-integer, and many (x, y) can map to the same (x', y'),
> reverse
> mapping is used. That is at every destination pixel (x', y'), we go
> and find
> the original pixel location (x, y) by taking (x, y) = f_1(x', y'),
> where f_1
> is the inverse function of f.
>
> Suppose the inverse function f_1 is given. So, for each pixel in the
> destination image, we can map to a non-integer pixel in the source
> image.
> Therefore, we must bilinear interpolate at the source location for the
> color.

> The transform(MESH) operation might be what you need.
> 
>     http://effbot.org/tag/PIL.Image.Image.transform
> 
> Instead of mapping between individual pixels, cover the output image
> with a grid, and calculate the source pixels for the corners of each
> grid rectangle (forming a quadrilateral area).  Pass in the resulting
> list of (grid rectangle, quadrilateral) pairs to transform(MESH), and
> it'll take care of the rest.
> 
> (you can use a generator to produce mesh pairs on the fly)
> 
> The result is (usually) an approximation, with the maximum error
> controlled by the grid resolution.

That's exactly what we're trying to do here.

- cover output image with a grid
- calculate source pixels for corners of each grid rectangle -->
  quadrilateral
- pass rect, quad pairs to MESH

Each rect is described by the *edges* (left, upper, right, lower),
i.e. the upper left and lower right corners.

Each quad is described by the *corners* (upper left, lower left, lower right, upper right).

    :::python
    im = Image.open('test.jpg')

    def get_rects(im):
        """Put a grid over an image and get the coords
        of the corners of each grid box."""
        res = 8
        r = res - 1
        rects = [(i, j, i, j + r, i + r, j + r, i + r, j) 
                        for i in range(0, w, res) for j in range(0, h, res)]
        return rects

    def trans(rect):
        """Find the corresponding source coords of the
        corners of each given rect."""
        ul = rect[0:2]
        ll = rect[2:4]
        lr = rect[4:6]
        ur = rect[6:8]
        P = (ul, ll, lr, ur)
        Quad = [f(p, dim, corr) for p in P]
        # quad = itertools.chain(Quad) 
        quad = sum(Quad, [])
        return quad
         
    rects = get_rects(im)
    quads = ((rect, trans(rect)) for rect in rects)
    imc = im.transform(im.size, MESH, data, BILINEAR)

