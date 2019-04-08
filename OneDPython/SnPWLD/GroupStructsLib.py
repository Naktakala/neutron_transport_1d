import numpy as np

def MakeSimple(num_groups):
  grpbounds=np.linspace(1.0,num_groups+1.0,num_groups)
  return grpbounds

def GetFromPDTdata(filename):
    pdtfile = open(filename,'r')
    auxlines = pdtfile.readlines()
    pdtfile.seek(0, 0)

    # ================== Read number of grps
    line = auxlines[3]
    words = line.split()
    G = int(words[5])
    grpbounds = np.zeros(G)

    # ================== Read group structure
    line_num = 16
    g = -1
    while (g<G):
        line_num+=1
        line = auxlines[line_num]
        words = line.split()

        for word in words:
            g+=1
            grpbounds[g] = float(word)
            if (g == (G - 1)):
                break

        if (g==(G-1)):
            break

    pdtfile.close()

    return grpbounds




