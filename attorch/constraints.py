
def positive(weight):
    weight.data *= weight.data.ge(0).float()

def negative(weight):
    weight.data *= weight.data.le(0).float()

def positive_except_self(weight):
    pos = weight.data.ge(0).float()
    if pos.size()[2] % 2 == 0 or pos.size()[3] % 2 == 0:
        raise ValueError('kernel size must be odd')
    ii, jj = pos.size()[2] // 2, pos.size()[3] // 2
    for i in range(pos.size()[0]):
        pos[i, i, ii, jj] = 1
    weight.data *= pos

