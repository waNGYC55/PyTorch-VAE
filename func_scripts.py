import os, sys

#file IO
def file2list(filename):
    r=[]
    with open(filename) as f:
        for line in f.readlines():
            r.append(line.split())
    return r

#pytorch analysis
def get_parameter_number(model):
    total_num=sum(p.numel() for p in model.parameters())
    trainable_num=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Memory':total_num*4/1000/1024}

def get_element_memory(element):
    return element.element_size()* element.nelement()/1000/1024