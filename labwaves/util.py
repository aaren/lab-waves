# Some useful functions that I use a lot.
import sys
import os
import errno
from collections import namedtuple
import cPickle as pickle
import json

import numpy as np

from config import data_dir

def pull_col(i, tsv, delim='\t'):
    # extract column i from a tsv file as a list
    f = open(tsv)
    f.seek(0)
    lines = f.read().split('\n')
    f.close()
    line_entries = []
    col = []
    for j in range(len(lines)):
        line_entries.append(lines[j].split(delim))
        col.append(line_entries[j][i])
    return col

def pull_line(j, tsv, delim='\t'):
    # pull line j from a tsv file as a list
    f = open(tsv)
    f.seek(0)
    lines = f.read().split('\n')
    f.close()
    line = lines[j].split(delim)
    return line

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

def write_data(data_dict, filename):
    """uses pickle to write a dict to disk for persistent storage"""
    output = open(filename, 'wb') # b is binary
    pickle.dump(data_dict, output, protocol=-1)
    output.close()

def read_data(filename):
    """reads in a dict from pickled file and returns it"""
    input = open(filename, 'rb')
    data_dict = pickle.load(input)
    input.close()
    return data_dict

def write_simple(run, data):
    """Writes out a JSON file for the given run and
    using the given data.
    """
    dataf = data_dir + 'simple/simple_%s.json' % run
    fout = open(dataf, 'w')
    fout.write(json.dumps(data))
    fout.close()

def read_simple(run, args='x, z, t'):
    """Reads in a JSON file that is in the format

    {keys: [[x,z,t], [x,z,t], ...]}

    and returns a data structure in the format

    {keys: [key(x=a, z=b, t=c), ...]}

    i.e. a dict of lists of namedtuples. If the
    namedtuple attributes are different from
    'x, z, t', these can be given as an argument
    but you must know a priori what these are as
    the JSON container doesn't store these.
    """
    dataf = data_dir + 'simple/simple_%s.json' % run
    fin = open(dataf, 'r')
    idata = json.loads(fin.read())
    fin.close()
    ndata = {}
    for k in idata:
        point = namedtuple(k, args)
        ndata[k] = [point(*p) for p in idata[k]]
    return ndata

def get_parameters(run, paramf, delim=None):
    headers = ['run_index', 'h_1/H', 'rho_0', 'rho_1', 'rho_2', 'alpha', 'D/H']
    types = ['S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']
    d = np.genfromtxt(paramf, dtype=types, names=headers)
    run_params = d[np.where(d['run_index'] == run)]
    parameters = dict(zip(headers, run_params.item()))
    return parameters

def cprint(string):
    print string,"\r",
    sys.stdout.flush()
def makedirs_p(path):
    """Emulate mkdir -p. Doesn't throw error if directory
    already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

