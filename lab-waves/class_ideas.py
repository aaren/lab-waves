

class Camera(object):
    pass

cam1 = Camera()
cam2 = Camera()

cam1.rulers=[]
cam2.rulers=[]

cam1.offsets

cam1.centre

# determine camera from path
# splitting -> string, e.g. 'cam1'
cam = 'cam1'

# how to use this to access object?
cam.offsets?

# is there a way to destring the string?


def obj_dic(d):
    """a useful method for turning a dict into an object, so that
    d['blah']['bleh'] is the same as d.blah.bleh.
    will work with any level of nesting inside the dict.
    """
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i, type(j)(obj_dic(sj)\
                    if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

