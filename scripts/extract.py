import sys

import labwaves

# runs with no synchronisation problems

runs = [
    'r13_01_08a',
    'r13_01_08c',
    'r13_01_08d',
    'r13_01_09a',
    'r13_01_11b',
    'r13_01_11c',
    'r13_01_12a',
]

for index in runs:
    print "extracting ", index, "... ",
    sys.stdout.flush()
    r = labwaves.Run(index)
    r.extract()
    r.save()
    print "done"
