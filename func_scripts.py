import os, sys

#file IO
def file2list(filename):
    r=[]
    with open(filename) as f:
        for line in f.readlines():
            r.append(line.split())
    return r

def file2dict(filename):
    r={}
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            r[line[0]]=line[1]
    return r

def list2file(l, filepath):
    with open(filepath, 'w') as f:
        for line in l:
            f.write(' '.join(line)+'\n')

#pytorch analysis
def get_parameter_number(model):
    total_num=sum(p.numel() for p in model.parameters())
    trainable_num=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Memory (MB)':total_num*4/1000/1024}

def get_element_memory(element):
    return element.element_size()* element.nelement()/1000/1024