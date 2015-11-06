import numpy
import random
import matplotlib.pyplot

learning_times=2500000
alpha=0.1

global_error_list=[]

def logsig(n):
    from math import e
    return float(1/(1+e**(-n)))

promoter_list=[]
not_promoter_list=[]

for i in open('promoter_next.th'):
    promoter_list.append(i.strip())
for i in open('not_promoter_next.th'):
    not_promoter_list.append(i.strip())

def compute_global_error():
    global_error=0.0
    for i in promoter_list:
        P_input=numpy.mat(i.strip().split(';')[0:-1]).T
        A_hidden=W_hidden*P_input+B_hidden
        for j in range(6):
            A_hidden.put(j,logsig(float(A_hidden.take(j).getA())))
        A_output=W_output*A_hidden+B_output
        for j in range(1):
            A_output.put(j,logsig(float(A_output.take(j).getA())))
        e=1-float(A_output.take(0).getA())
        global_error=global_error+e**2
    for i in not_promoter_list:
        P_input=numpy.mat(i.strip().split(';')[0:-1]).T
        A_hidden=W_hidden*P_input+B_hidden
        for j in range(6):
            A_hidden.put(j,logsig(float(A_hidden.take(j).getA())))
        A_output=W_output*A_hidden+B_output
        for j in range(1):
            A_output.put(j,logsig(float(A_output.take(j).getA())))
        e=0-float(A_output.take(0).getA())
        global_error=global_error+e**2
    return global_error/(2*len(promoter_list)+len(not_promoter_list))

W_hidden=numpy.mat(numpy.ones((6,24)))
for i in range(6*24):
    W_hidden.put(i,random.uniform(-1, 1))
B_hidden=numpy.mat(numpy.ones((6,1)))
for i in range(6*1):
    B_hidden.put(i,random.uniform(-1, 1))
W_output=numpy.mat(numpy.ones((1,6)))
for i in range(1*6):
    W_output.put(i,random.uniform(-1, 1))
B_output=numpy.mat(numpy.ones((1,1)))
for i in range(1*1):
    B_output.put(i,random.uniform(-1, 1))

global_error_list.append(compute_global_error())

for i in range(learning_times):
    #print('Learning times: '+str(i+1))
    if random.randint(0,1)==1:
        seq_input=promoter_list[random.randint(0,len(promoter_list)-1)]
        expected_value_t=1.0
    else:
        seq_input=not_promoter_list[random.randint(0,len(not_promoter_list)-1)]
        expected_value_t=0.0

    P_input=numpy.mat([float(x) for x in seq_input.strip().split(';')[0:-1]]).T

    A_hidden=W_hidden*P_input+B_hidden
    for i in range(6):
        A_hidden.put(i,logsig(float(A_hidden.take(i).getA())))
    A_output=W_output*A_hidden+B_output
    for i in range(1):
        A_output.put(i,logsig(float(A_output.take(i).getA())))

    s_output=-2*(expected_value_t-float(A_output.take(0).getA()))*(1-float(A_output.take(0).getA()))*float(A_output.take(0).getA())

    A_square=numpy.mat(numpy.zeros((6,6)))
    for i in range(6):
        A_square.put(i*(6+1),float(A_hidden.take(i).getA())*(1-float(A_hidden.take(i).getA())))
    s_hidden=float(s_output)*A_square*W_output.T

    W_output=W_output-alpha*s_output*A_hidden.T
    B_output=B_output-alpha*s_output
    W_hidden=W_hidden-alpha*s_hidden*P_input.T
    B_hidden=B_hidden-alpha*s_hidden

    global_error_list.append(compute_global_error())

fout_global_error=open('global_error_final.th','w')
for i in global_error_list:
    fout_global_error.write(str(i)+"\n")
fout_global_error.close()

fout_W_hidden=open('W_hidden_final.th','w')
for i in range(6*24):
    fout_W_hidden.write(str(float(W_hidden.take(i).getA()))+"\n")
fout_W_hidden.close()

fout_B_hidden=open('B_hidden_final.th','w')
for i in range(6*1):
    fout_B_hidden.write(str(float(B_hidden.take(i).getA()))+"\n")
fout_B_hidden.close()

fout_W_output=open('W_output_final.th','w')
for i in range(1*6):
    fout_W_output.write(str(float(W_output.take(i).getA()))+"\n")
fout_W_output.close()

fout_B_output=open('B_output_final.th','w')
for i in range(1*1):
    fout_B_output.write(str(float(B_output.take(i).getA()))+"\n")
fout_B_output.close()

matplotlib.pyplot.plot(range(0,learning_times+1),global_error_list)
matplotlib.pyplot.title("The convergence curve of global error")
matplotlib.pyplot.xlabel("Learning times")
matplotlib.pyplot.ylabel("Global error")
matplotlib.pyplot.savefig('global_error_final.pdf')

