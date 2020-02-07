#!/usr/bin/python
# Naive LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.models import Model
from keras.models import load_model
import general_uni
import json
import sys
from keras.callbacks import ModelCheckpoint
import controller_evl as con_eval
import numpy as np

def gendata(pre_network,rnn_output,pre_acc,acc,k1,k2):
    rnn2_data=[]
    for net in range(len(rnn_output)):
        temp_net={"feature":rnn_output[net].tolist(),"score":1}#gen.score(pre_network,rnn_output[net],pre_acc,acc[net],k1,k2)}
        rnn2_data.append(temp_net)
    return rnn2_data

#X,y =general_uni.gen_trainingData() #pad+1
X=[]
y=[]
network=[[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]]]]
pre_acc=87
file="round1.json"
X1,y1 =general_uni.gen_trainingData_score(network,pre_acc,file,2)
X.extend(X1)
y.extend(y1)


# file="round1.json"
# X1,y1 =general_uni.gen_trainingData(file,0)
# X.extend(X1)
# y.extend(y1)
#
# file="round0.json"
# X2,y2 =general_uni.gen_trainingData(file,1)
# X.extend(X2)
# y.extend(y2)
#X_test,name=general_uni.gen_testData("round2_new")
#print len(X)



#X_test = numpy.reshape(X_test, (len(X_test),-1,61))
print len(X)
#X_train=X[30:100]
X_train=X[50:]
#X_train.extend(X[120:])
#y_train=y[30:100]
y_train=y[50:]
#y_train.extend(y[120:])

print len(X_train)
#X_val=X[0:30]
X_val=X[0:50]
#X_val.extend(X[100:120])
#y_val=y[0:30]
y_val=y[0:50]
#y_val.extend(y[100:120])
X_train = numpy.reshape(X_train, (len(X_train),-1,61))
X_val = numpy.reshape(X_val, (len(X_val),-1,61))



model = Sequential()
model.add(LSTM(100, input_shape=(X[1].shape[0],X[1].shape[1])))
# model.add(GRU(100, dropout_W=0.3,dropout_U=0.3,input_shape=(X[1].shape[0],X[1].shape[1]),name='rnn1',return_sequences=True))
# model.add(GRU(100, dropout_W=0.3,dropout_U=0.3,name='rnn2',return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

filepath='model/rnn37/rnn2.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]

model.fit(X_train, y_train, nb_epoch=10000, batch_size=64, callbacks=callbacks_list,verbose=2,validation_data=(X_val,y_val),shuffle=True)
sys.exit(0)

# model.save('model/rnn/rnn1.h5')
# summarize performance of the model
#predic=model.predict(X_va,verbose=0l)
#scores = model.evaluate(X, y, verbose=0)
#print("Model Accuracy: %.2f%%" % (scores[1]*100))


round='round1'
data_path='data_controller/'+round+'.json'
file_path=open(data_path,"r")
s=json.load(file_path)
X_test=[]
y_name=[]
y_acc=[]

for i in s:

    network=[]

    for b in i['feature']:
        block_temp=[]
        for uni in b:
            uni_temp=[]
            for op in uni:
                #print np.array(uni).flatten().tolist()
                #network.append(np.array(uni).flatten().tolist())
                uni_temp.extend(np.array(op).flatten().tolist())
            block_temp.append(uni_temp)
        network.extend(block_temp)

    X_test.append(network)


    #X_test.append(i['feature'])
    y_name.append(i['net'])
    y_acc.append(i['acc'])
#X_test,name=general_uni.gen_testData(round)
results=[]

for i in range(len(X_test)):
    result_temp=[y_acc[i],y_name[i]]
    results.append(result_temp)

results=sorted(results,key=lambda x:x[0],reverse=True)
results_real_150=[]
for i in range(150):
    results_real_150.append(results[i][1])
results_real=[]
results_real_100=[]
for i in range(100):
    results_real_100.append(results[i][1])
results_real_50=[]
for i in range(50):
    results_real_50.append(results[i][1])

results_real=[]
for i in range(len(results)):
    results_real.append(results[i][1])

X_test = numpy.reshape(X_test, (len(X_test),-1,61))
model=load_model('model/rnn/rnn0_g_ls.h5')
model_output=model.predict(X_test)
result=[]
for i in range(len(model_output)):
    result_temp=[model_output[i][0],y_name[i]]
    result.append(result_temp)

result_sort=sorted(result,key=lambda x:x[0],reverse=True)

results_score_150=[]
for i in range(150):
    results_score_150.append(result_sort[i][1])
results_score_100=[]
for i in range(100):
    results_score_100.append(result_sort[i][1])
results_score_50=[]
for i in range(50):
    results_score_50.append(result_sort[i][1])

eval_k=50
#calculate the map
result_map=[]
for r in results_score_150:
    if r in results_real_150:
        result_map.append(1)
    else:
        result_map.append(0)

print 'AP150: '+ str(con_eval.average_precision(result_map))
result_map=[]
for r in results_score_100:
    if r in results_real_100:
        result_map.append(1)
    else:
        result_map.append(0)

print 'AP100: '+ str(con_eval.average_precision(result_map))
result_map=[]

for r in results_score_50:
    if r in results_real_50:
        result_map.append(1)
    else:
        result_map.append(0)

print 'AP50: '+ str(con_eval.average_precision(result_map))
result_ncdg=[]
for i in result_sort:
    result_ncdg.append(1/float(results_real.index(i[1])+1))


print 'ndcg50 :'+str(con_eval.ndcg_at_k(result_ncdg,50))
print 'ndcg100 :'+str(con_eval.ndcg_at_k(result_ncdg,100))
print 'ndcg150 :'+str(con_eval.ndcg_at_k(result_ncdg,150))
sys.exit(0)













model=load_model('model/rnn/rnn1_score.h5')
round='round2'
X_test,name=general_uni.gen_testData(round)
X_test = numpy.reshape(X_test, (len(X_test),-1,61))

model_output=model.predict(X_test)
print len(model_output)


for i in range(len(model_output)):
    result_temp=[model_output[i][0],name[i]]
    result.append(result_temp)

result_sort=sorted(result,key=lambda x:x[0],reverse=True)



output_path="pre_result/rnn_round6_sorted_score_73_2.json"
result_json=[]
for r in result_sort:
    result_json.append({"acc":float(r[0]), "net":r[1]})

data_object=open(output_path,"w")
json.dump(result_json,data_object)
data_object.close()

output_path="pre_result/rnn_round6_sorted_score_100_73_2.json"
result_json=[]
for r in range(100):
    result_json.append({"net":result_sort[r][1]})

data_object=open(output_path,"w")
json.dump(result_json,data_object)
data_object.close()


sys.exit(0)

data_path="rnn2.json"
rnn_model=Model(input=model.input, output=model.get_layer('rnn1').output)


rnn_output=rnn_model.predict(X)
# 15 39 56 84 93
total_data=[]
#round5
#round 1
pre_network=[[[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0]],]
rnn2_data=gendata(pre_network,rnn_output[93:107],87,y[93:107],0.5,0.5)


data_object=open(data_path,"w")
json.dump(rnn2_data,data_object)
data_object.close()
result_object=open(data_path,"r")
s=json.load(result_object)
print s



# demonstrate some model predictions
'''
for pattern in dataX:
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print seq_in, "->", result
'''
