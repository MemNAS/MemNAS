#!/usr/bin/python
# -*- coding: UTF-8 -*-

#from graphviz import Digraph
import copy
import os
import numpy as np
import numpy
import f_to_n_uni as f2n
import sys
def draw_network(network,name,path):
    net =Digraph (comment='network',format='png')
    net.node("s","start")
    net.node("e","end")
    max_out=100
    for i in range(len(network)):
        if network[i][1].index(1)<max_out:
            max_out=network[i][1].index(1)
        if network[i][2].index(1)==7:    #1x7 then 7x1 conv
            net.node(str(i),"1x7 then 7x1")
        if network[i][2].index(1)==6:    #1x3 then 3x1 conv
            net.node(str(i),"1x3 then 3x1")
        if network[i][2].index(1)==5:    #3x3 dilated convolution
            net.node(str(i),"3x3 dilated")
        if network[i][2].index(1)==4:    #avgpool
            net.node(str(i),"avgpool")
        if network[i][2].index(1)==3:    #avgpool
            net.node(str(i),"maxpool")
        if network[i][2].index(1)==2:    #1x1 conv layer
            net.node(str(i),"1x1 conv")
        if network[i][2].index(1)==1:     #3x3 conv layer
            net.node(str(i),"3x3 conv")
        if network[i][2].index(1)==0:
            net.node(str(i),"3x3 depthwise-separable ")

    for l in range(len(network)):
        if network[l][0]==[0,0,0,0,0,0,0,1]:
            net.edge("s",str(l))
        if network[l][1].index(1)==max_out:
            net.edge(str(l),"e")

        for k in range(l,len(network)):

            if network[l][1]==network[k][0]:
                #print("1")
                net.edge(str(l),str(k))
    if not os.path.exists(path):
        os.mkdir(path)


    net.render(path+"/"+str(name))

def ordernetwork(networks):
    return (sorted(networks, key = lambda net: (-net[0].index(1),-net[1].index(1))))

def increase_location(layer, length):
    index=layer.index(1)
    layer=[0 for i in range(length)]
    layer[index-1]=1
    #index=layer[1].index(1)
    #layer[1]=[0,0,0,0,0,0,0,0]
    #layer[1][index-1]=1
    return layer

def decrase_location(layer,length):
    index=layer.index(1)
    layer=[0 for i in range(length)]
    layer[index+1]=1
    #index=layer[1].index(1)
    #layer[1]=[0,0,0,0,0,0,0,0]
    #layer[1][index-1]=1
    return layer

def generate_big(ori_network_feature):

    Inp_location=[]
    Out_location=[]
    Opera=[]
    for layer in ori_network_feature:
        Inp_location.append(layer[0])
        Out_location.append(layer[1])
        Opera.append(layer[2])
    #print (Inp_location)
    #print (Out_location)
    #print (Opera)

    #generate one new layer
    candidate_layers=[]
    candidate_networks=[]


    #generate parallel layers
    for inp in Inp_location:
        for ou in Out_location:
            if inp.index(1)>ou.index(1):
                for i in range(8):
                    op=[0,0,0,0,0,0,0,0]
                    op[7-i]=1
                    #print (op)
                    tmp_layer=[]
                    tmp_layer.append(inp)
                    tmp_layer.append(ou)
                    tmp_layer.append(op)

                    if tmp_layer not in candidate_layers:
                        candidate_layers.append(tmp_layer)

    for cl in candidate_layers:
        tmp=copy.deepcopy(ori_network_feature)
        tmp.append(cl)
        candidate_networks.append(ordernetwork(tmp))

    '''
    #generate special parallel layers
    for inp in Inp_location:
        for ou in Out_location:
            count=[i for i,val in enumerate(Inp_location) if val==ou]
            if len(count)>1 :
                tmp=copy.deepcopy(ori_network_feature)
                for c_l in count:



    for cl in candidate_layers:
        tmp=copy.deepcopy(ori_network_feature)
        tmp.append(cl)
        candidate_networks.append(ordernetwork(tmp))
    '''


    #generate sequential layers after the current layer
    for inp in range(len(Inp_location)):
        temp_network=copy.deepcopy(ori_network_feature)
        new_inp=Out_location[inp]
        #print (new_inp)
        new_out=increase_location(new_inp)
        for l in range(len(temp_network)):
            if l!=inp:
                if temp_network[l][0].index(1)<=new_inp.index(1):
                    temp_network[l][0]=increase_location(temp_network[l][0])
                    #temp_network[l][1]=increase_location(temp_network[l][1])
                if temp_network[l][1].index(1)<=new_inp.index(1):
                    temp_network[l][1]=increase_location(temp_network[l][1])
                    #temp_network[l][1]=increase_location(temp_network[l][1])
        for i in range(8):
            ttmp_network=copy.deepcopy(temp_network)
            op=[0,0,0,0,0,0,0,0]
            op[7-i]=1
            #print (op)
            tmp_layer=[]
            tmp_layer.append(new_inp)
            tmp_layer.append(new_out)
            tmp_layer.append(op)
            ttmp_network.append(tmp_layer)
            if ordernetwork(ttmp_network) not in candidate_networks:
                candidate_networks.append(ordernetwork(ttmp_network))


    return candidate_networks

def generate_small(ori_network_feature):
    Inp_location=[]
    Out_location=[]
    Opera=[]
    for layer in ori_network_feature:
        Inp_location.append(layer[0])
        Out_location.append(layer[1])
        Opera.append(layer[2])
    candidate_networks=[]
    flag=0

    for out in range(len(Out_location)):
        temp_network=copy.deepcopy(ori_network_feature)
        for out_other in range(len(Out_location)):
            if Out_location[out_other]==Out_location[out] and out!=out_other:
                if Inp_location[out_other]==Inp_location[out]:
                    del temp_network[out]
                    flag=1
                    break

        if flag==1:
            flag=0

        else:

            for net_other in range(len(temp_network)):


                if temp_network[net_other][0].index(1)<temp_network[out][0].index(1):
                    temp_network[net_other][0]=decrase_location(temp_network[net_other][0])

                if temp_network[net_other][1].index(1)<temp_network[out][1].index(1):
                    temp_network[net_other][1]=decrase_location(temp_network[net_other][1])

            del temp_network[out]
        candidate_networks.append(temp_network)
    return candidate_networks


class Network():
    def __init__(self,network,max_location):
        self.network=network
        self.block_feature_1=network[0]
        self.block_feature_2=network[1]
        self.block_feature_3=network[2]
        self.block_feature_4=network[3]
        self.block_feature_5=network[4]
        self.max_location=max_location
        self.operation=7
        self.link=1
        self.INP_ini= [0 for i in range(self.max_location)]
        self.INP_ini[-1]=1
        self.OUT_ini= [0 for i in range(self.max_location)]
        self.OUT_ini[-2]=1
        self.blobk1_channel=32
        self.blobk2_channel=32
        self.blobk3_channel=32
        self.blobk4_channel=32
        self.blobk5_channel=32



        def output_channel(block, num):
            Out_location=[]
            link=[]
            for layer in block:
                Out_location.append(layer[2])
                link.append(layer[5])
            temp=len(Out_location[0])
            for o in Out_location:
                if o.index(1)<temp:
                    temp=o.index(1)
            max=temp
            output_count_temp=0
            for n in range(len(Out_location)):
                #if Out_location[n].index(1)==max:
                if link[n].index(1)==0:
                    output_count_temp+=1

            return output_count_temp*num


        def _gen_initial_networks():
            networks=[]
            for o1 in range(self.operation):
                op1=[0 for i in range(self.operation)]
                op1[o1]=1
                for o2 in range(self.operation):
                    op2=[0 for i in range(self.operation)]
                    op2[o2]=1
                    for l in range(self.link):
                        temp_link=[0 for i in range(self.link)]
                        temp_link[l]=1
                        new_cell=[]
                        new_cell.append(self.INP_ini)
                        new_cell.append(self.INP_ini)
                        new_cell.append(self.OUT_ini)
                        new_cell.append(op1)
                        new_cell.append(op2)
                        new_cell.append(temp_link)

                        new_block=[]
                        new_block.append(new_cell)

                        new_network=[]
                        new_network.append(new_block)
                        new_network.append(new_block)
                        new_network.append(new_block)
                        new_network.append(new_block)
                        new_network.append(new_block)
                        networks.append(new_network)
            return networks

        if len(self.block_feature_1)==0:
            self.round_0_networks=_gen_initial_networks()

        else:
            self.input_location=[self.INP_ini]
            for b in self.block_feature_1:
                self.input_location.append(b[2])
            self.output_location=increase_location(self.block_feature_1[-1][2], self.max_location)


        self.block1_output=output_channel(self.block_feature_1,self.blobk1_channel)
        self.block2_output=output_channel(self.block_feature_2,self.blobk2_channel)
        self.block3_output=output_channel(self.block_feature_3,self.blobk3_channel)
        self.block4_output=output_channel(self.block_feature_4,self.blobk4_channel)
        self.block5_output=output_channel(self.block_feature_5,self.blobk5_channel)


    def gen_big(self):
        networks=[]

        for o1 in range(self.operation):
            op1=[0 for i in range(self.operation)]
            op1[o1]=1
            for o2 in range(self.operation):
                op2=[0 for i in range(self.operation)]
                op2[o2]=1
                for l in range(self.link):
                    temp_link=[0 for i in range(2)]
                    temp_link[l]=1
                    for inp1 in self.input_location:
                        for inp2 in self.input_location:
                            block_feature_1=copy.deepcopy(self.block_feature_1)
                            block_feature_2=copy.deepcopy(self.block_feature_2)
                            block_feature_3=copy.deepcopy(self.block_feature_3)
                            block_feature_4=copy.deepcopy(self.block_feature_4)
                            block_feature_5=copy.deepcopy(self.block_feature_5)

                            new_cell=[]
                            new_cell.append(inp1)
                            new_cell.append(inp2)
                            new_cell.append(self.output_location)
                            new_cell.append(op1)
                            new_cell.append(op2)
                            new_cell.append(temp_link)

                            block_feature_1.append(new_cell)
                            block_feature_2.append(new_cell)
                            block_feature_3.append(new_cell)
                            block_feature_4.append(new_cell)
                            block_feature_5.append(new_cell)

                            new_network=[]
                            new_network.append(block_feature_1)
                            new_network.append(block_feature_2)
                            new_network.append(block_feature_3)
                            new_network.append(block_feature_4)
                            new_network.append(block_feature_5)
                            networks.append(new_network)
        return networks


    def remove(self):
        networks=[]
        block_feature_1=copy.deepcopy(self.block_feature_1)
        block_feature_2=copy.deepcopy(self.block_feature_2)
        block_feature_3=copy.deepcopy(self.block_feature_3)
        block_feature_4=copy.deepcopy(self.block_feature_4)
        block_feature_5=copy.deepcopy(self.block_feature_5)


        for uni in range(len(self.block_feature_1)):
            if self.block_feature_1[uni][3] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_1)
                block_feature[uni][3]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(block_feature)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)
            if self.block_feature_1[uni][4] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_1)
                block_feature[uni][4]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(block_feature)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(self.block_feature_2)):
            if self.block_feature_2[uni][3] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_2)
                block_feature[uni][3]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(block_feature)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)
            if self.block_feature_2[uni][4] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_2)
                block_feature[uni][4]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(block_feature)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(self.block_feature_3)):
            if self.block_feature_3[uni][3] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_3)
                block_feature[uni][3]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(block_feature)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)
            if self.block_feature_3[uni][4] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_3)
                block_feature[uni][4]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(block_feature)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)


        for uni in range(len(self.block_feature_4)):
            if self.block_feature_4[uni][3] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_4)
                block_feature[uni][3]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(block_feature)
                new_network.append(self.block_feature_5)
                networks.append(new_network)
            if self.block_feature_4[uni][4] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_4)
                block_feature[uni][4]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(block_feature)
                new_network.append(self.block_feature_5)
                networks.append(new_network)


        for uni in range(len(self.block_feature_5)):
            if self.block_feature_5[uni][3] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_5)
                block_feature[uni][3]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(block_feature)
                networks.append(new_network)
            if self.block_feature_5[uni][4] !=[0 for i in range(self.operation)]:
                block_feature=copy.deepcopy(self.block_feature_5)
                block_feature[uni][4]=[0 for i in range(self.operation)]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(block_feature)
                networks.append(new_network)

        return networks

    def remove_fin(self):
        networks=[]
        block_feature_1=copy.deepcopy(self.block_feature_1)
        block_feature_2=copy.deepcopy(self.block_feature_2)
        block_feature_3=copy.deepcopy(self.block_feature_3)
        block_feature_4=copy.deepcopy(self.block_feature_4)
        block_feature_5=copy.deepcopy(self.block_feature_5)
        input_location_1=[x[0] for x in block_feature_1]
        input_location_1.extend([x[1] for x in block_feature_1])

        input_location_2=[x[0] for x in block_feature_2]
        input_location_2.extend([x[1] for x in block_feature_2])

        input_location_3=[x[0] for x in block_feature_3]
        input_location_3.extend([x[1] for x in block_feature_3])

        input_location_4=[x[0] for x in block_feature_4]
        input_location_4.extend([x[1] for x in block_feature_4])

        input_location_5=[x[0] for x in block_feature_5]
        input_location_5.extend([x[1] for x in block_feature_5])
        for uni in range(len(block_feature_1)):
            block_feature=copy.deepcopy(self.block_feature_1)
            if block_feature[uni][2] in input_location_1:
                block_feature[uni][5]=[0,1]
                new_network=[]
                new_network.append(block_feature)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(block_feature_2)):
            block_feature=copy.deepcopy(self.block_feature_2)
            if block_feature[uni][2] in input_location_2:
                block_feature[uni][5]=[0,1]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(block_feature)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(block_feature_3)):
            block_feature=copy.deepcopy(self.block_feature_3)
            if block_feature[uni][2] in input_location_3:
                block_feature[uni][5]=[0,1]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(block_feature)
                new_network.append(self.block_feature_4)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(block_feature_4)):
            block_feature=copy.deepcopy(self.block_feature_4)
            if block_feature[uni][2] in input_location_4:
                block_feature[uni][5]=[0,1]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(block_feature)
                new_network.append(self.block_feature_5)
                networks.append(new_network)

        for uni in range(len(block_feature_5)):
            block_feature=copy.deepcopy(self.block_feature_5)
            if block_feature[uni][2] in input_location_5:
                block_feature[uni][5]=[0,1]
                new_network=[]
                new_network.append(self.block_feature_1)
                new_network.append(self.block_feature_2)
                new_network.append(self.block_feature_3)
                new_network.append(self.block_feature_4)
                new_network.append(block_feature)
                networks.append(new_network)


        return networks



    def cal_weight(self):

        weight_count=0
        for i in self.block_feature_1:
            if i[0].index(1)==6:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=3*3*3*self.blobk1_channel
                    if i[3].index(1)==5:
                        temp=3*3*3*3+3*1*self.blobk1_channel
                    if i[3].index(1)==4:
                        temp=3*5*5+3*1*self.blobk1_channel
                    if i[3].index(1)==3:
                        temp=3*7*self.blobk1_channel+self.blobk1_channel*7*self.blobk1_channel
                    if i[3].index(1)==2:
                        temp=3*3*3*self.blobk1_channel
                    if i[3].index(1)==1:
                        temp=3*self.blobk1_channel
                    if i[3].index(1)==0:
                        temp=3*self.blobk1_channel
                else:
                    temp=0
            else:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.blobk1_channel*3*3*self.blobk1_channel
                    if i[3].index(1)==5:
                        temp=self.blobk1_channel*3*3+self.blobk1_channel*1*self.blobk1_channel
                    if i[3].index(1)==4:
                        temp=self.blobk1_channel*5*5+self.blobk1_channel*1*self.blobk1_channel
                    if i[3].index(1)==3:
                        temp=self.blobk1_channel*7*self.blobk1_channel+self.blobk1_channel*7*self.blobk1_channel
                    if i[3].index(1)==2:
                        temp=self.blobk1_channel*3*3*self.blobk1_channel
                    if i[3].index(1)==1:
                        temp=self.blobk1_channel*self.blobk1_channel
                    if i[3].index(1)==0:
                        temp=self.blobk1_channel*self.blobk1_channel
                else:
                    temp=0
            weight_count+=temp
            if i[1].index(1)==6:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=3*3*3*self.blobk1_channel
                    if i[4].index(1)==5:
                        temp=3*3*3+3*1*self.blobk1_channel
                    if i[4].index(1)==4:
                        temp=3*5*5+3*1*self.blobk1_channel
                    if i[4].index(1)==3:
                        temp=3*7*self.blobk1_channel+self.blobk1_channel*7*self.blobk1_channel
                    if i[4].index(1)==2:
                        temp=3*3*3*self.blobk1_channel
                    if i[4].index(1)==1:
                        temp=self.blobk1_channel*self.blobk1_channel
                    if i[4].index(1)==0:
                        temp=self.blobk1_channel*self.blobk1_channel
                else:
                    temp=0
            else:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.blobk1_channel*3*3*self.blobk1_channel
                    if i[4].index(1)==5:
                        temp=self.blobk1_channel*3*3+self.blobk1_channel*1*self.blobk1_channel
                    if i[4].index(1)==4:
                        temp=self.blobk1_channel*5*5+self.blobk1_channel*1*self.blobk1_channel
                    if i[4].index(1)==3:
                        temp=self.blobk1_channel*7*self.blobk1_channel+self.blobk1_channel*7*self.blobk1_channel
                    if i[4].index(1)==2:
                        temp=self.blobk1_channel*3*3*self.blobk1_channel
                    if i[4].index(1)==1:
                        temp=self.blobk1_channel*self.blobk1_channel
                    if i[4].index(1)==0:
                        temp=self.blobk1_channel*self.blobk1_channel
                else:
                    temp=0
            weight_count+=temp

        for i in self.block_feature_2:
            if i[0].index(1)==6:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.block1_output*3*3*self.blobk2_channel
                    if i[3].index(1)==5:
                        temp=self.block1_output*3*3+self.block1_output*1*self.blobk2_channel
                    if i[3].index(1)==4:
                        temp=self.block1_output*5*5+self.block1_output*1*self.blobk2_channel
                    if i[3].index(1)==3:
                        temp=self.block1_output*7*self.blobk2_channel+self.blobk2_channel*7*self.blobk2_channel
                    if i[3].index(1)==2:
                        temp=self.block1_output*3*3*self.blobk2_channel
                    if i[3].index(1)==1:
                        temp=self.block1_output*self.blobk2_channel
                    if i[3].index(1)==0:
                        temp=self.block1_output*self.blobk2_channel
                else:
                    temp=0
            else:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.blobk2_channel*3*3*self.blobk2_channel
                    if i[3].index(1)==5:
                        temp=self.blobk2_channel*3*3+self.blobk2_channel*1*self.blobk2_channel
                    if i[3].index(1)==4:
                        temp=self.blobk2_channel*5*5+self.blobk2_channel*1*self.blobk2_channel
                    if i[3].index(1)==3:
                        temp=self.blobk2_channel*7*self.blobk2_channel+self.blobk2_channel*7*self.blobk2_channel
                    if i[3].index(1)==2:
                        temp=self.blobk2_channel*3*3*self.blobk2_channel
                    if i[3].index(1)==1:
                        temp=self.blobk2_channel*self.blobk2_channel
                    if i[3].index(1)==0:
                        temp=self.blobk2_channel*self.blobk2_channel
                else:
                    temp=0
            weight_count+=temp
            if i[1].index(1)==6:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.block1_output*3*3*self.blobk2_channel
                    if i[4].index(1)==5:
                        temp=self.block1_output*3*3+self.block1_output*1*self.blobk2_channel
                    if i[4].index(1)==4:
                        temp=self.block1_output*5*5+self.block1_output*1*self.blobk2_channel
                    if i[4].index(1)==3:
                        temp=self.block1_output*7*self.blobk2_channel+self.blobk2_channel*7*self.blobk2_channel
                    if i[4].index(1)==2:
                        temp=self.block1_output*3*3*self.blobk2_channel
                    if i[4].index(1)==1:
                        temp=self.block1_output*self.blobk2_channel
                    if i[4].index(1)==0:
                        temp=self.block1_output*self.blobk2_channel
                else:
                    temp=0
            else:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.blobk2_channel*3*3*self.blobk2_channel
                    if i[4].index(1)==5:
                        temp=self.blobk2_channel*3*3+self.blobk2_channel*1*self.blobk2_channel
                    if i[4].index(1)==4:
                        temp=self.blobk2_channel*5*5+self.blobk2_channel*1*self.blobk2_channel
                    if i[4].index(1)==3:
                        temp=self.blobk2_channel*7*self.blobk2_channel+self.blobk2_channel*7*self.blobk2_channel
                    if i[4].index(1)==2:
                        temp=self.blobk2_channel*3*3*self.blobk2_channel
                    if i[4].index(1)==1:
                        temp=self.blobk2_channel*self.blobk2_channel
                    if i[4].index(1)==0:
                        temp=self.blobk2_channel*self.blobk2_channel
                else:
                    temp=0
            weight_count+=temp

        for i in self.block_feature_3:
            if i[0].index(1)==6:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.block2_output*3*3*self.blobk3_channel
                    if i[3].index(1)==5:
                        temp=self.block2_output*3*3+self.block2_output*1*self.blobk3_channel
                    if i[3].index(1)==4:
                        temp=self.block2_output*5*5+self.block2_output*1*self.blobk3_channel
                    if i[3].index(1)==3:
                        temp=self.block2_output*7*self.blobk3_channel+self.blobk3_channel*7*self.blobk3_channel
                    if i[3].index(1)==2:
                        temp=self.block2_output*3*3*self.blobk3_channel
                    if i[3].index(1)==1:
                        temp=self.block2_output*self.blobk3_channel
                    if i[3].index(1)==0:
                        temp=self.block2_output*self.blobk3_channel
                else:
                    temp=0
            else:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.blobk3_channel*3*3*self.blobk3_channel
                    if i[3].index(1)==5:
                        temp=self.blobk3_channel*3*3+self.blobk3_channel*1*self.blobk3_channel
                    if i[3].index(1)==4:
                        temp=self.blobk3_channel*5*5+self.blobk3_channel*1*self.blobk3_channel
                    if i[3].index(1)==3:
                        temp=self.blobk3_channel*7*self.blobk3_channel+self.blobk3_channel*7*self.blobk3_channel
                    if i[3].index(1)==2:
                        temp=self.blobk3_channel*3*3*self.blobk3_channel
                    if i[3].index(1)==1:
                        temp=self.blobk3_channel*self.blobk3_channel
                    if i[3].index(1)==0:
                        temp=self.blobk3_channel*self.blobk3_channel
                else:
                    temp=0
            weight_count+=temp
            if i[1].index(1)==6:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.block2_output*3*3*self.blobk3_channel
                    if i[4].index(1)==5:
                        temp=self.block2_output*3*3+self.block2_output*1*self.blobk3_channel
                    if i[4].index(1)==4:
                        temp=self.block2_output*5*5+self.block2_output*1*self.blobk3_channel
                    if i[4].index(1)==3:
                        temp=self.block2_output*7*self.blobk3_channel+self.blobk3_channel*7*self.blobk3_channel
                    if i[4].index(1)==2:
                        temp=self.block2_output*3*3*self.blobk3_channel
                    if i[4].index(1)==1:
                        temp=self.block2_output*self.blobk3_channel
                    if i[4].index(1)==0:
                        temp=self.block2_output*self.blobk3_channel
                else:
                    temp=0
            else:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.blobk3_channel*3*3*self.blobk3_channel
                    if i[4].index(1)==5:
                        temp=self.blobk3_channel*3*3+self.blobk3_channel*1*self.blobk3_channel
                    if i[4].index(1)==4:
                        temp=self.blobk3_channel*5*5+self.blobk3_channel*1*self.blobk3_channel
                    if i[4].index(1)==3:
                        temp=self.blobk3_channel*7*self.blobk3_channel+self.blobk3_channel*7*self.blobk3_channel
                    if i[4].index(1)==2:
                        temp=self.blobk3_channel*3*3*self.blobk3_channel
                    if i[4].index(1)==1:
                        temp=self.blobk3_channel*self.blobk3_channel
                    if i[4].index(1)==0:
                        temp=self.blobk3_channel*self.blobk3_channel
                else:
                    temp=0
            weight_count+=temp



        for i in self.block_feature_4:
            if i[0].index(1)==6:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.block3_output*3*3*self.blobk4_channel
                    if i[3].index(1)==5:
                        temp=self.block3_output*3*3+self.block3_output*1*self.blobk4_channel
                    if i[3].index(1)==4:
                        temp=self.block3_output*5*5+self.block3_output*1*self.blobk4_channel
                    if i[3].index(1)==3:
                        temp=self.block3_output*7*self.blobk4_channel+self.blobk4_channel*7*self.blobk4_channel
                    if i[3].index(1)==2:
                        temp=self.block3_output*3*3*self.blobk4_channel
                    if i[3].index(1)==1:
                        temp=self.block3_output*self.blobk4_channel
                    if i[3].index(1)==0:
                        temp=self.block3_output*self.blobk4_channel
                else:
                    temp=0
            else:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.blobk4_channel*3*3*self.blobk4_channel
                    if i[3].index(1)==5:
                        temp=self.blobk4_channel*3*3+self.blobk4_channel*1*self.blobk4_channel
                    if i[3].index(1)==4:
                        temp=self.blobk4_channel*5*5+self.blobk4_channel*1*self.blobk4_channel
                    if i[3].index(1)==3:
                        temp=self.blobk4_channel*7*self.blobk4_channel+self.blobk4_channel*7*self.blobk4_channel
                    if i[3].index(1)==2:
                        temp=self.blobk4_channel*3*3*self.blobk4_channel
                    if i[3].index(1)==1:
                        temp=self.blobk4_channel*self.blobk4_channel
                    if i[3].index(1)==0:
                        temp=self.blobk4_channel*self.blobk4_channel
                else:
                    temp=0
            weight_count+=temp
            if i[1].index(1)==6:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.block3_output*3*3*self.blobk4_channel
                    if i[4].index(1)==5:
                        temp=self.block3_output*3*3+self.block3_output*1*self.blobk4_channel
                    if i[4].index(1)==4:
                        temp=self.block3_output*5*5+self.block3_output*1*self.blobk4_channel
                    if i[4].index(1)==3:
                        temp=self.block3_output*7*self.blobk4_channel+self.blobk4_channel*7*self.blobk4_channel
                    if i[4].index(1)==2:
                        temp=self.block3_output*3*3*self.blobk4_channel
                    if i[4].index(1)==1:
                        temp=self.block3_output*self.blobk4_channel
                    if i[4].index(1)==0:
                        temp=self.block3_output*self.blobk4_channel
                else:
                    temp=0
            else:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.blobk4_channel*3*3*self.blobk4_channel
                    if i[4].index(1)==5:
                        temp=self.blobk4_channel*3*3+self.blobk4_channel*1*self.blobk4_channel
                    if i[4].index(1)==4:
                        temp=self.blobk4_channel*5*5+self.blobk4_channel*1*self.blobk4_channel
                    if i[4].index(1)==3:
                        temp=self.blobk4_channel*7*self.blobk4_channel+self.blobk4_channel*7*self.blobk4_channel
                    if i[4].index(1)==2:
                        temp=self.blobk4_channel*3*3*self.blobk4_channel
                    if i[4].index(1)==1:
                        temp=self.blobk4_channel*self.blobk4_channel
                    if i[4].index(1)==0:
                        temp=self.blobk4_channel*self.blobk4_channel
                else:
                    temp=0
            weight_count+=temp

        for i in self.block_feature_5:
            if i[0].index(1)==6:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.block4_output*3*3*self.blobk5_channel
                    if i[3].index(1)==5:
                        temp=self.block4_output*3*3+self.block4_output*1*self.blobk5_channel
                    if i[3].index(1)==4:
                        temp=self.block4_output*5*5+self.block4_output*1*self.blobk5_channel
                    if i[3].index(1)==3:
                        temp=self.block4_output*7*self.blobk5_channel+self.blobk5_channel*7*self.blobk5_channel
                    if i[3].index(1)==2:
                        temp=self.block4_output*3*3*self.blobk5_channel
                    if i[3].index(1)==1:
                        temp=self.block4_output*self.blobk5_channel
                    if i[3].index(1)==0:
                        temp=self.block4_output*self.blobk5_channel
                else:
                    temp=0
            else:
                if 1 in i[3]:
                    if i[3].index(1)==6:
                        temp=self.blobk5_channel*3*3*self.blobk5_channel
                    if i[3].index(1)==5:
                        temp=self.blobk5_channel*3*3+self.blobk5_channel*1*self.blobk5_channel
                    if i[3].index(1)==4:
                        temp=self.blobk5_channel*5*5+self.blobk5_channel*1*self.blobk5_channel
                    if i[3].index(1)==3:
                        temp=self.blobk5_channel*7*self.blobk5_channel+self.blobk5_channel*7*self.blobk5_channel
                    if i[3].index(1)==2:
                        temp=self.blobk5_channel*3*3*self.blobk5_channel
                    if i[3].index(1)==1:
                        temp=self.blobk5_channel*self.blobk5_channel
                    if i[3].index(1)==0:
                        temp=self.blobk5_channel*self.blobk5_channel
                else:
                    temp=0
            weight_count+=temp
            if i[1].index(1)==6:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.block4_output*3*3*self.blobk5_channel
                    if i[4].index(1)==5:
                        temp=self.block4_output*3*3+self.block4_output*1*self.blobk5_channel
                    if i[4].index(1)==4:
                        temp=self.block4_output*5*5+self.block4_output*1*self.blobk5_channel
                    if i[4].index(1)==3:
                        temp=self.block4_output*7*self.blobk5_channel+self.blobk5_channel*7*self.blobk5_channel
                    if i[4].index(1)==2:
                        temp=self.block4_output*3*3*self.blobk5_channel
                    if i[4].index(1)==1:
                        temp=self.block4_output*self.blobk5_channel
                    if i[4].index(1)==0:
                        temp=self.block4_output*self.blobk5_channel
                else:
                    temp=0
            else:
                if 1 in i[4]:
                    if i[4].index(1)==6:
                        temp=self.blobk5_channel*3*3*self.blobk5_channel
                    if i[4].index(1)==5:
                        temp=self.blobk5_channel*3*3+self.blobk5_channel*1*self.blobk5_channel
                    if i[4].index(1)==4:
                        temp=self.blobk5_channel*5*5+self.blobk5_channel*1*self.blobk5_channel
                    if i[4].index(1)==3:
                        temp=self.blobk5_channel*7*self.blobk5_channel+self.blobk5_channel*7*self.blobk5_channel
                    if i[4].index(1)==2:
                        temp=self.blobk5_channel*3*3*self.blobk5_channel
                    if i[4].index(1)==1:
                        temp=self.blobk5_channel*self.blobk5_channel
                    if i[4].index(1)==0:
                        temp=self.blobk5_channel*self.blobk5_channel
                else:
                    temp=0
            weight_count+=temp
        return weight_count
    def ini_mul_channel(self,ope,ini,channel1,channel2):
        score=0
        for i in ini:
            if len(ope[i-1]['input'])==1:
                if ope[i-1]['input'][0]==0:
                    score+=channel1
                    score+=channel2
                else:
                    score+=channel2
                    score+=channel2
            else:
                if ope[i-1]['input'][0]==0:
                    score+=channel1
                    score+=channel2
                else:
                    score+=channel2
                    score+=channel2

                if ope[i-1]['input'][1]==0:
                    score+=channel1
                    score+=channel2
                else:
                    score+=channel2
                    score+=channel2
        return score


    def cal_in(self):
        ope=[]
        count=1
        for uni in self.block_feature_1:
            input1=len(uni[0])-uni[0].index(1)
            if uni[0].index(1)==len(uni[0])-1:
                input1=0
            else:
                input1=(len(uni[0])-uni[0].index(1)-1)*3
            node_temp1={"no":count,"input":[input1],"output":[count+2],"state":0}
            ope.append(node_temp1)
            count+=1
            if uni[1].index(1)==len(uni[1])-1:
                input2=0
            else:
                input2=(len(uni[1])-uni[1].index(1)-1)*3
            node_temp2={"no":count,"input":[input2],"output":[count+1],"state":0}
            ope.append(node_temp2)
            count+=1
            if uni[5]==[1,1]:
                out=[]
            else:
                out=[0,]
            node_temp3={"no":count,"input":[count-2,count-1],"output":out,"state":0}
            ope.append(node_temp3)
            count+=1

        for o in ope:
            for temp in ope:
                if [o['no']]==temp['input']:
                    o['output'].append(temp['no'])
        #print ope
        begin=0
        max_ini=0
        ini_temp=[]
        for node in ope:
            if node['input']==[begin]:
                node['state']=1
                ini_temp.append(node['no'])

        score=self.ini_mul_channel(ope,ini_temp,self.block1_output,self.blobk2_channel)*32*32
        max_ini=score

        for n in range(10):
            ope_temp=copy.deepcopy(ope)
            for node in ope_temp:
                if len(node['input'])==1:
                    if ope[node['input'][0]-1]['state']==1:

                        node['state']=1
                        ini_temp.append(node['no'])
                else:
                    if ope[node['input'][0]-1]['state']==1 and ope[node['input'][1]-1]['state']==1:
                        #print 1
                        node['state']=1
                        ini_temp.append(node['no'])

            score=self.ini_mul_channel(ope_temp,ini_temp,self.block1_output,self.blobk2_channel)*32*32
            # print score

            if max_ini<score:
                max_ini=score


            ope=copy.deepcopy(ope_temp)
            for node in ope:
                if node['state']==1:
                    if len(node['output'])==1 and node['output'][0]!=0:
                        if ope_temp[node['output'][0]-1]['state']==1:
                            node['state']=2
                            ini_temp.remove(node['no'])
                    else:
                        if len(node['output'])==1:
                            node['state']=2
                        else:
                            #print node['output']
                            if ope_temp[node['output'][1]-1]['state']==1:
                                node['state']=2


        return max_ini




if __name__=="__main__":
    #round 0
    network=[[],[],[],[],[]]
    '''

    #round 0 best result: net 53 acc 87%

    network=[[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]]]

    net=Network(network,15)
    print (net.cal_weight())
    print (net.cal_in())


    #print len(net.remove_fin())
    # sys.exit(0)
    candi=[]
    candi.extend(net.gen_big())
    candi.extend(net.remove())
    candi.extend(net.remove_fin())

    cot=1
    for n in candi:
        #print n[4][5]
        f2n.feature_to_network(n,"networks/round0_acc/","net"+str(cot))
        #draw_network(n,"net"+str(cot),"network_figure/round4_new")
        cot+=1
