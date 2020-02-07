#!/usr/bin/python
import re
import sys
import json
import os
import numpy as np
import generate_uni
#def search_acc:
#    re_network=".*for(.*)is.*"

def search_file(round):
    result_path="data/"+round+".json"
    root="./"
    #search the log file
    data=[]
    log_pattern= re.compile(round+".*"+".log")
    for rt, dirs, files in os.walk(root):
        for d in files:
            match = log_pattern.match(d)
            if match:
                print d
                data.extend(generate_data(d,round))
                #print len(generate_data(d,round))
        break

    result_object=open(result_path,"w")
    json.dump(data,result_object)
    result_object.close()
    result_object=open(result_path,"r")
    s=json.load(result_object)
    #for k in s:
    #    print k['acc']
    print len(s)



def generate_data(log_file,round):
    file_path=log_file
    file_object=open(file_path)
    data=[]
    for lines in file_object.readlines():
        re_network=".*fornet(.*)is.*"
        re_acc1=".*tensor\((.*), device='cuda:0'\)"
        re_acc2=".*is:(.*)"
        if re.findall(re_network,lines):
            net=re.findall(re_network,lines)
            acc1=re.findall(re_acc1,lines)
            acc2=re.findall(re_acc2,lines)
            print net[0]
            file_path_temp=("networks/"+round+"/"+str(net[0]+'.py'))
            file_object_temp=open(file_path_temp)
            net_feature=0
            for l in file_object_temp.readlines():
                re_net_feature=".*#network feature: (.*)"
                if re.findall(re_net_feature,l):
                    net_feature=re.findall(re_net_feature,l)
                    break
            file_object_temp.close()
            network_feature=json.loads(net_feature[0])
            if acc1:
                #print acc1
                network={"feature":network_feature, "acc":float(acc1[0])}
            if (not acc1) and acc2:
                network={"feature":network_feature, "acc":float(acc2[0])}
            data.append(network)
            #print network_feature
            #print float(acc[0])
    return data

def search_for_con(round):
    result_path="data_controller/"+round+".json"
    root="./"
    #search the log file
    data=[]
    log_pattern= re.compile(round+".*"+".log")
    for rt, dirs, files in os.walk(root):
        for d in files:
            match = log_pattern.match(d)
            if match:
                print d
                data.extend(gen_controller_eval(d,round))
                #print len(generate_data(d,round))
        break

    result_object=open(result_path,"w")
    json.dump(data,result_object)
    result_object.close()
    result_object=open(result_path,"r")
    s=json.load(result_object)
    #for k in s:
    #    print k['acc']
    print len(s)
def gen_controller_eval(log_file,round):
    file_path=log_file
    file_object=open(file_path)
    data=[]
    for lines in file_object.readlines():
        re_network=".*fornet(.*)is.*"
        re_acc1=".*tensor\((.*), device='cuda:0'\)"
        re_acc2=".*is:(.*)"
        if re.findall(re_network,lines):
            net=re.findall(re_network,lines)
            acc1=re.findall(re_acc1,lines)
            acc2=re.findall(re_acc2,lines)
            #print net[0]
            file_path_temp=("networks/"+round+"/net"+str(net[0]+'.py'))
            file_object_temp=open(file_path_temp)
            net_feature=0
            for l in file_object_temp.readlines():
                re_net_feature=".*#network feature: (.*)"
                if re.findall(re_net_feature,l):
                    net_feature=re.findall(re_net_feature,l)
                    break
            file_object_temp.close()
            network_feature=json.loads(net_feature[0])
            if acc1:
                #print acc1
                network={"feature":network_feature, "net":net[0], "acc":float(acc1[0])}
            if (not acc1) and acc2:
                network={"feature":network_feature,"net":net[0], "acc":float(acc2[0])}
            data.append(network)
            #print network_feature
            #print float(acc[0])
    return data











def generate_testdata(log_file,round):
    file_path=log_file
    #print file_path

    file_object_temp=open(file_path)
    net_feature=0
    for l in file_object_temp.readlines():
        re_net_feature=".*#network feature: (.*)"
        if re.findall(re_net_feature,l):
            net_feature=re.findall(re_net_feature,l)
            break
    file_object_temp.close()
    network_feature=json.loads(net_feature[0])

            #print network_feature
            #print float(acc[0])
    return network_feature
def generate_test_data(round):
    result_path="pre/"+round+".json"
    root="networks/"+round+"/"
    #search the log file
    data=[]
    log_pattern= re.compile("net"+".*"+".py")
    for rt, dirs, files in os.walk(root):
        for d in files:
            match = log_pattern.match(d)
            if match:
                #print d
                network_feature=generate_testdata(root+d,round)
                #print len(generate_data(d,round))
                network={"feature":network_feature, "name":str(d)}
                data.append(network)


    result_object=open(result_path,"w")
    json.dump(data,result_object)
    result_object.close()
    result_object=open(result_path,"r")
    s=json.load(result_object)
    print len(s)

def arrange_arr(result_arr,result_nets,acc,i):
    #print result_nets
    #print i

    for l in range(len(result_arr)-1,i,-1):
        result_arr[l]=result_arr[l-1]
        result_nets[l]=result_nets[l-1]
    #print result_nets
    return result_arr,result_nets



def find_best_network(round,path):

    log_pattern= re.compile(round+".*"+".log")
    num=30
    result_arr=[0 for i in range(num)]
    result_nets=[0 for i in range(num)]
    for rt, dirs, files in os.walk(path):
        for d in files:
            match = log_pattern.match(d)
            if match:
                print d
                file_path=path+d
                file_object=open(file_path)
                for lines in file_object.readlines():
                    re_network=".*fornet(.*)is.*"
                    re_acc1=".*tensor\((.*), device='cuda:0'\)"
                    re_acc2=".*is:(.*)"

                    if re.findall(re_network,lines):

                        net=re.findall(re_network,lines)
                        acc1=re.findall(re_acc1,lines)
                        acc2=re.findall(re_acc2,lines)
                        #print float(acc2[0])

                        if acc1:
                            for i in range(len(result_arr)):
                                if float(acc1[0])>result_arr[i]:
                                    #print net[0]
                                    #result_arr,result_nets=
                                    arrange_arr(result_arr,result_nets,acc1[0],i)

                                    result_arr[i]=float(acc1[0])
                                    result_nets[i]=net[0]
                                    #print result_nets
                                    break

                        if (not acc1) and acc2:
                            for i in range(len(result_arr)):
                                if float(acc2[0])>result_arr[i]:
                                    #print net[0]
                                    #result_arr,result_nets=

                                    arrange_arr(result_arr,result_nets,acc2[0],i)
                                    result_arr[i]=float(acc2[0])
                                    result_nets[i]=net[0]

                                    break
        break

    file_path="networks/"+round+"/"
    #print result_arr, result_nets
    end=[]
    for e in range(num):
        temp=[]
        temp.append(result_nets[e])
        temp.append(result_arr[e])
        if result_nets[e]!=0:
            feature=generate_testdata(file_path+str(result_nets[e])+".py",round)
            temp.append(generate_uni.Network(feature,15).cal_weight())
            temp.append(generate_uni.Network(feature,15).cal_in())
            temp.append(generate_uni.Network(feature,15).cal_in()+generate_uni.Network(feature,15).cal_weight())
            end.append(temp)
    return end
    #return result_net1,max_score1,generate_uni.Network(feature1,15).cal_weight(),generate_uni.Network(feature1,15).cal_in(),result_net2,max_score2,generate_uni.Network(feature2,15).cal_weight(),generate_uni.Network(feature2,15).cal_in(),result_net3,max_score3,generate_uni.Network(feature3,15).cal_weight(),generate_uni.Network(feature3,15).cal_in(),


def gen_trainingData(file,pad_extra):
    training_data=[]
    training_label=[]
    for rt, dirs, files in os.walk("data"):
        for f in files:
            with open(rt+'/'+f) as load_f:
                if f==file:
                    print(f)
                    load_dict=json.load(load_f)
                    #print load_dict
                    for n in load_dict:
                        network=[]

                        for b in n['feature']:
                            block_temp=[]
                            for uni in b:
                                uni_temp=[]
                                for op in uni:
                                    #print np.array(uni).flatten().tolist()
                                    #network.append(np.array(uni).flatten().tolist())
                                    uni_temp.extend(np.array(op).flatten().tolist())
                                block_temp.append(uni_temp)
                            network.extend(block_temp)

                        training_data.append(network)
                        training_label.append(n['acc'])
                #print(len(training_data))
    #print np.array(training_data)
    #print np.array(training_label)
    #training_data_tmp=padding_data(np.array(training_data))
    training_data_tmp=padding_training_data((training_data),pad_extra)
    return training_data_tmp, (training_label)

def gen_trainingData_score(pre_network,pre_acc,file,pad_extra):
    training_data=[]
    training_label=[]
    for rt, dirs, files in os.walk("data"):
        for f in files:
            with open(rt+'/'+f) as load_f:

                if f==file:
                    print(f)
                    load_dict=json.load(load_f)
                    #print load_dict
                    for n in load_dict:
                        network=[]

                        for b in n['feature']:
                            block_temp=[]
                            for uni in b:
                                uni_temp=[]
                                for op in uni:
                                    #print np.array(uni).flatten().tolist()
                                    #network.append(np.array(uni).flatten().tolist())
                                    uni_temp.extend(np.array(op).flatten().tolist())
                                block_temp.append(uni_temp)
                            network.extend(block_temp)
                        #print network
                        #os.exit()
                        #print np.array(network)[0]
                        #network=np.array(network)
                        #print training_data
                        training_data.append(network)
                        training_label.append(score(pre_network,n['feature'],pre_acc,n['acc'],0.7,0.3)*100)

                #print(len(training_data))
    #print np.array(training_data)
    #print np.array(training_label)
    #training_data_tmp=padding_data(np.array(training_data))
    training_data_tmp=padding_training_data((training_data),pad_extra)
    return training_data_tmp, (training_label)

def gen_testData(file_path):
    test_data=[]
    test_name=[]
    result_object=open("pre/"+file_path+".json","r")
    load_dict=json.load(result_object)
    #print load_dict
    for n in load_dict:
        network=[]

        for b in n['feature']:
            block_temp=[]
            for uni in b:
                uni_temp=[]
                for op in uni:
                    #print np.array(uni).flatten().tolist()
                    #network.append(np.array(uni).flatten().tolist())
                    uni_temp.extend(np.array(op).flatten().tolist())
                block_temp.append(uni_temp)
            network.extend(block_temp)

        test_data.append(network)
        test_name.append(n['name'])
    test_data=padding_data((test_data))

    return test_data, test_name
def cal_int(network):
    Inp_location=[]
    Out_location=[]
    Opera=[]
    for layer in network:
        Inp_location.append(layer[0])
        Out_location.append(layer[1])
        Opera.append(layer[2])
    max_ini=0
    for l in Inp_location:
        count=0
        count=find_end(l, Inp_location,Out_location, count)
        #print count
        #print count
        temp_ini=count
        if temp_ini>max_ini:
            max_ini=temp_ini

    return count

def score(pre_network,network,pre_acc,acc,k1,k2):
    acc_nor=(acc-pre_acc)/float(pre_acc)

    mem_wei=(generate_uni.Network(pre_network,15).cal_weight()-generate_uni.Network(network,15).cal_weight())/float(generate_uni.Network(pre_network,15).cal_weight())
    mem_in=(generate_uni.Network(pre_network,15).cal_in()-generate_uni.Network(network,15).cal_in())/float(generate_uni.Network(pre_network,15).cal_in())
    k3=0.3
    k4=0.7

    return k1*(acc_nor)+k2*(k3*mem_wei+k4*mem_in)

def padding_data(data):

    max_layer=0
    for n in data:
        #print len(n)
        if len(n)>max_layer:
            max_layer=len(n)
    #print max_layer
    #os.exit()
    for n in range(len(data)):
        if len(data[n])<max_layer:
            pad_count=(max_layer-len(data[n]))/5

            insert_loc=len(data[n])/5
            insert_distant=len(data[n])/5
            insert_uni=[]
            for i in range(pad_count):
                temp=np.zeros((61,),dtype=np.int)
                temp=np.array([temp])
                insert_uni.extend(temp)

                #print temp
            #data[n]=np.r_[data[n],temp]

            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            #insert_loc=insert_loc+pad_count+insert_loc
            #np.insert(data[n],insert_loc,insert_uni,0)
            data[n]=np.append(data[n],insert_uni,0)

    return data

def padding_training_data(data,pad_extra):

    max_layer=0
    for n in data:
        #print len(n)
        if len(n)>max_layer:
            max_layer=len(n)
    #print max_layer
    #os.exit()
    for n in range(len(data)):
        #if len(data[n])<max_layer:
        pad_count=(max_layer-len(data[n]))/5+pad_extra


        insert_loc=len(data[n])/5
        insert_distant=len(data[n])/5
        insert_uni=[]
        for i in range(pad_count):
            temp=np.zeros((61,),dtype=np.int)
            temp=np.array([temp])
            insert_uni.extend(temp)
            #print temp
        #data[n]=np.r_[data[n],temp]
        if pad_count:
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            insert_loc=insert_loc+pad_count+insert_distant
            data[n]=np.insert(data[n],insert_loc,insert_uni,0)
            #insert_loc=insert_loc+pad_count+insert_loc
            #np.insert(data[n],insert_loc,insert_uni,0)
            data[n]=np.append(data[n],insert_uni,0)

    return data

def find_end(location, Inp_location,Out_location, count):
    #if location in Out_location:
    num_o=[i for i,val in enumerate(Out_location) if val==location]
    if len(num_o)<2:
        #count+=1
        return 1
    else:
        temp=0

        for o in num_o:
            kk=find_end(Inp_location[o], Inp_location,Out_location, count)
            temp+=kk
            #print temp
        return temp
    return 0

def sb(file_path):
    test_data=[]
    test_name=[]
    result_object=open("pre_acc/"+file_path+".json","r")
    load_dict=json.load(result_object)
    #print load_dict
    wei_arr=[0 for i in range(20)]
    in_arr=[0 for i in range(20)]
    result_nets=[0 for i in range(20)]
    for n in load_dict:
        network=[]

        # for b in n['feature']:
        #     block_temp=[]
        #     for uni in b:
        #         uni_temp=[]
        #         for op in uni:
        #             #print np.array(uni).flatten().tolist()
        #             #network.append(np.array(uni).flatten().tolist())
        #             uni_temp.extend(np.array(op).flatten().tolist())
        #         block_temp.append(uni_temp)
        #     network.extend(block_temp)
        mem_wei=generate_uni.Network(n['feature'],15).cal_weight()
        mem_in=generate_uni.Network(n['feature'],15).cal_in()
        #test_name.append(n['name'])

        for i in range(len(wei_arr)):
            if float(mem_wei)>wei_arr[i]:

                arrange_sb(wei_arr,result_nets,in_arr,mem_wei,i)

                wei_arr[i]=float(mem_wei)
                result_nets[i]=n['name']
                in_arr[i]=float(mem_in)
                #print result_nets
                break
    return wei_arr,result_nets,in_arr
def arrange_sb(result_arr,result_nets,ini,acc,i):
    #print result_nets
    #print i

    for l in range(len(result_arr)-1,i,-1):
        result_arr[l]=result_arr[l-1]
        result_nets[l]=result_nets[l-1]
        ini[l]=ini[l-1]
    #print result_nets
    return result_arr,result_nets,ini

def generate_acc_data(round):
    result_path="pre_acc/"+round+".json"
    root="networks/"+round+"/"
    #search the log file
    data=[]
    log_pattern= re.compile("net"+".*"+".py")
    for rt, dirs, files in os.walk(root):
        for d in files:
            match = log_pattern.match(d)
            if match:
                #print d
                network_feature=generate_testdata(root+d,round)
                #print len(generate_data(d,round))
                network={"feature":network_feature, "name":str(d)}
                data.append(network)


    result_object=open(result_path,"w")
    json.dump(data,result_object)
    result_object.close()
    result_object=open(result_path,"r")
    s=json.load(result_object)
    print len(s)
#print s
if __name__=="__main__":
    #gen_trainingData()
    # generate_acc_data("round6_acc")
    #gen_testData("round2_new")
    #search_file("round5")
    #search_for_con("round1")
    '''
    test_network=[[[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0]]]
    '''


    #print cal_int(test_network)#*32*32*64*2#+32*32*64
    #print cal_wei(test_network)
    #cal_int
