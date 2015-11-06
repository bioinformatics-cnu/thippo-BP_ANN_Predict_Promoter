seq_sorting_list=['ATCG', 'ATGC', 'ACTG', 'ACGT', 'AGTC', 'AGCT', 'TACG', 'TAGC', 'TCAG', 'TCGA', 'TGAC', 'TGCA', 'CATG', 'CAGT', 'CTAG', 'CTGA', 'CGAT', 'CGTA', 'GATC', 'GACT', 'GTAC', 'GTCA', 'GCAT', 'GCTA']
promoter_filename='train_Tset_m1.th'
not_promoter_filename='train_Fset_m1.th'
learning_times=2500000
alpha=0.1
input_number_begin=324
hidden_number_begin=81
output_numberr_begin=1

import numpy
import random
import matplotlib.pyplot
import multiprocessing

def logsig(n):
    from math import e
    return float(1/(1+e**(-n)))

def seq_translate_list(templet,seq):
    final_list=[]
    templet=templet.strip();templet=templet.upper()
    seq=seq.strip();seq=seq.upper()
    for i in seq:
        for j in templet:
            if i in j:
                final_list.append(1.0)
            else:
                final_list.append(-1.0)
    return final_list

promoter_list=[]
not_promoter_list=[]

for i in open(promoter_filename):
    promoter_list.append(i.strip())
for i in open(not_promoter_filename):
    not_promoter_list.append(i.strip())

def multi_ANN_BP(templet):

    def compute_global_error():
        global_error=0.0
        for i in promoter_list:
            P_input=numpy.mat(seq_translate_list(templet,i)).T
            A_hidden=W_hidden*P_input+B_hidden
            for j in range(hidden_number_begin):
                A_hidden.put(j,logsig(float(A_hidden.take(j).getA())))
            A_output=W_output*A_hidden+B_output
            for j in range(output_numberr_begin):
                A_output.put(j,logsig(float(A_output.take(j).getA())))
            e=1-float(A_output.take(0).getA())
            global_error=global_error+e**2
        for i in not_promoter_list:
            P_input=numpy.mat(seq_translate_list(templet,i)).T
            A_hidden=W_hidden*P_input+B_hidden
            for j in range(hidden_number_begin):
                A_hidden.put(j,logsig(float(A_hidden.take(j).getA())))
            A_output=W_output*A_hidden+B_output
            for j in range(output_numberr_begin):
                A_output.put(j,logsig(float(A_output.take(j).getA())))
            e=0-float(A_output.take(0).getA())
            global_error=global_error+e**2
        return global_error/(2*len(promoter_list)+len(not_promoter_list))

    W_hidden=numpy.mat(numpy.ones((hidden_number_begin,input_number_begin)))
    for i in range(hidden_number_begin*input_number_begin):
        W_hidden.put(i,random.uniform(-1, 1))
    B_hidden=numpy.mat(numpy.ones((hidden_number_begin,output_numberr_begin)))
    for i in range(hidden_number_begin*1):
        B_hidden.put(i,random.uniform(-1, 1))
    W_output=numpy.mat(numpy.ones((output_numberr_begin,hidden_number_begin)))
    for i in range(output_numberr_begin*hidden_number_begin):
        W_output.put(i,random.uniform(-1, 1))
    B_output=numpy.mat(numpy.ones((1,1)))
    for i in range(output_numberr_begin*output_numberr_begin):
        B_output.put(i,random.uniform(-1, 1))

    global_error_list=[]

    global_error_list.append(compute_global_error())

    for i in range(learning_times):
        #print('Learning times of '+templet+': '+str(i+1))
        if random.randint(0,1)==1:
            seq_input=promoter_list[random.randint(0,len(promoter_list)-1)]
            expected_value_t=1.0
        else:
            seq_input=not_promoter_list[random.randint(0,len(not_promoter_list)-1)]
            expected_value_t=0.0
		
        P_input=numpy.mat(seq_translate_list(templet,seq_input)).T

        A_hidden=W_hidden*P_input+B_hidden
        for j in range(hidden_number_begin):
            A_hidden.put(j,logsig(float(A_hidden.take(j).getA())))
        A_output=W_output*A_hidden+B_output
        for j in range(output_numberr_begin):
            A_output.put(j,logsig(float(A_output.take(j).getA())))

        s_output=-2*(expected_value_t-float(A_output.take(0).getA()))*(1-float(A_output.take(0).getA()))*float(A_output.take(0).getA())

        A_square=numpy.mat(numpy.zeros((hidden_number_begin,hidden_number_begin)))
        for j in range(hidden_number_begin):
            A_square.put(j*(hidden_number_begin+1),float(A_hidden.take(j).getA())*(1-float(A_hidden.take(j).getA())))
        s_hidden=float(s_output)*A_square*W_output.T

        W_output=W_output-alpha*s_output*A_hidden.T
        B_output=B_output-alpha*s_output
        W_hidden=W_hidden-alpha*s_hidden*P_input.T
        B_hidden=B_hidden-alpha*s_hidden

        global_error_list.append(compute_global_error())

    fout_global_error=open('global_error_'+templet+'.th','w')
    for i in global_error_list:
        fout_global_error.write(str(i)+"\n")
    fout_global_error.close()

    fout_W_hidden=open('W_hidden_'+templet+'.th','w')
    for i in range(hidden_number_begin*input_number_begin):
        fout_W_hidden.write(str(float(W_hidden.take(i).getA()))+"\n")
    fout_W_hidden.close()

    fout_B_hidden=open('B_hidden_'+templet+'.th','w')
    for i in range(hidden_number_begin*output_numberr_begin):
        fout_B_hidden.write(str(float(B_hidden.take(i).getA()))+"\n")
    fout_B_hidden.close()

    fout_W_output=open('W_output_'+templet+'.th','w')
    for i in range(output_numberr_begin*hidden_number_begin):
        fout_W_output.write(str(float(W_output.take(i).getA()))+"\n")
    fout_W_output.close()

    fout_B_output=open('B_output_'+templet+'.th','w')
    for i in range(output_numberr_begin*output_numberr_begin):
        fout_B_output.write(str(float(B_output.take(i).getA()))+"\n")
    fout_B_output.close()

    matplotlib.pyplot.plot(range(0,learning_times+1),global_error_list)
    matplotlib.pyplot.title("The convergence curve of global error")
    matplotlib.pyplot.xlabel("Learning times")
    matplotlib.pyplot.ylabel("Global error")
    matplotlib.pyplot.savefig('global_error_'+templet+'.pdf')
    matplotlib.pyplot.clf()

if __name__=='__main__':
    p = multiprocessing.Pool()
    for templet in seq_sorting_list:
        p.apply_async(multi_ANN_BP, args=(templet,))
    p.close()
    p.join()
