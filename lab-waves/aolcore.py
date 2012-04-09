# Some useful functions that I use a lot.
import pickle
import sys

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
    pickle.dump(data_dict, output)
    output.close()

def read_data(filename):
    """reads in a dict from pickled file and returns it"""
    input = open(filename, 'rb')
    data_dict = pickle.load(input)
    input.close()
    return data_dict

def get_parameters(run, paramf, delim=None): 
    p_runs = pull_col(0, paramf, delim) 
    run_params = pull_line(p_runs.index(run), paramf, delim)
    headers = pull_line(0, paramf, delim)
    parameters = dict(zip(headers, run_params))
    return parameters

def cprint(string):
    print string,"\r",
    sys.stdout.flush()
