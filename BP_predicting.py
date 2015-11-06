import numpy

seq_sorting_list=['ATCG', 'ATGC', 'ACTG', 'ACGT', 'AGTC', 'AGCT', 'TACG', 'TAGC', 'TCAG', 'TCGA', 'TGAC', 'TGCA', 'CATG', 'CAGT', 'CTAG', 'CTGA', 'CGAT', 'CGTA', 'GATC', 'GACT', 'GTAC', 'GTCA', 'GCAT', 'GCTA']
input_number_begin=324
hidden_number_begin=81
output_number_begin=1
input_number_final=24
hidden_number_final=6
output_number_final=1
testT_filename='predict_Tset_m1.th'
testF_filename='predict_Fset_m1.th'

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

W_hidden_final_list=[]
B_hidden_final_list=[]
W_output_final_list=[]
B_output_final_list=[]
for i in open('W_hidden_final.th'):
    W_hidden_final_list.append(i.strip())
for i in open('B_hidden_final.th'):
    B_hidden_final_list.append(i.strip())
for i in open('W_output_final.th'):
    W_output_final_list.append(i.strip())
for i in open('B_output_final.th'):
    B_output_final_list.append(i.strip())
W_hidden_final=numpy.mat(numpy.ones((hidden_number_final,input_number_final)))
for j in range(hidden_number_final*input_number_final):
    W_hidden_final.put(j, W_hidden_final_list[j])
B_hidden_final=numpy.mat(numpy.ones((hidden_number_final,output_number_final)))
for j in range(hidden_number_final*output_number_final):
    B_hidden_final.put(j,B_hidden_final_list[j])
W_output_final=numpy.mat(numpy.ones((output_number_final,hidden_number_final)))
for j in range(output_number_final*hidden_number_final):
    W_output_final.put(j,W_output_final_list[j])
B_output_final=numpy.mat(numpy.ones((output_number_final,output_number_final)))
for j in range(output_number_final*output_number_final):
    B_output_final.put(j,B_output_final_list[j])

for i in seq_sorting_list:
    locals()['W_hidden_'+i+'_list']=[]
    locals()['B_hidden_'+i+'_list']=[]
    locals()['W_output_'+i+'_list']=[]
    locals()['B_output_'+i+'_list']=[]
    for j in open('W_hidden_'+i+'.th'):
        locals()['W_hidden_'+i+'_list'].append(j.strip())
    for j in open('B_hidden_'+i+'.th'):
        locals()['B_hidden_'+i+'_list'].append(j.strip())
    for j in open('W_output_'+i+'.th'):
        locals()['W_output_'+i+'_list'].append(j.strip())
    for j in open('B_output_'+i+'.th'):
        locals()['B_output_'+i+'_list'].append(j.strip())
    locals()['W_hidden_'+i]=numpy.mat(numpy.ones((hidden_number_begin,input_number_begin)))
    for j in range(hidden_number_begin*input_number_begin):
        locals()['W_hidden_'+i].put(j, locals()['W_hidden_'+i+'_list'][j])
    locals()['B_hidden_'+i]=numpy.mat(numpy.ones((hidden_number_begin,output_number_begin)))
    for j in range(hidden_number_begin*output_number_begin):
        locals()['B_hidden_'+i].put(j,locals()['B_hidden_'+i+'_list'][j])
    locals()['W_output_'+i]=numpy.mat(numpy.ones((output_number_begin,hidden_number_begin)))
    for j in range(output_number_begin*hidden_number_begin):
        locals()['W_output_'+i].put(j,locals()['W_output_'+i+'_list'][j])
    locals()['B_output_'+i]=numpy.mat(numpy.ones((output_number_begin,output_number_begin)))
    for j in range(output_number_begin*output_number_begin):
        locals()['B_output_'+i].put(j,locals()['B_output_'+i+'_list'][j])

np_total=0;nn_total=0
np_true=0;nn_false=0

fout_predict=open('predict_result.th','w')
fout_correct=open('correct_result.th','w')

for i in open(testT_filename):
    np_total=np_total+1
    i=i.strip()
    matrix_list=[]
    for j in seq_sorting_list:
        P_input_begin=numpy.mat(seq_translate_list(j,i)).T
        A_hidden_begin=locals()['W_hidden_'+j]*P_input_begin+locals()['B_hidden_'+j]
        for k in range(hidden_number_begin):
            A_hidden_begin.put(k,logsig(float(A_hidden_begin.take(k).getA())))
        A_output_begin=locals()['W_output_'+j]*A_hidden_begin+locals()['B_output_'+j]
        for k in range(output_number_begin):
            A_output_begin.put(k,logsig(float(A_output_begin.take(k).getA())))
        matrix_list.append(float(A_output_begin.take(0).getA()))
    P_input_final=numpy.mat(matrix_list).T
    A_hidden_final=W_hidden_final*P_input_final+B_hidden_final
    for j in range(hidden_number_final):
        A_hidden_final.put(j,logsig(float(A_hidden_final.take(j).getA())))
    A_output_final=W_output_final*A_hidden_final+B_output_final
    for j in range(output_number_final):
        A_output_final.put(j,logsig(float(A_output_final.take(j).getA())))
    fout_predict.write(str(float(A_output_final.take(j).getA()))+'	'+'T'+'\n')
    if float(A_output_final.take(0).getA())>0.5:
        np_true=np_true+1

for i in open(testF_filename):
    nn_total=nn_total+1
    i=i.strip()
    matrix_list=[]
    for j in seq_sorting_list:
        P_input_begin=numpy.mat(seq_translate_list(j,i)).T
        A_hidden_begin=locals()['W_hidden_'+j]*P_input_begin+locals()['B_hidden_'+j]
        for k in range(hidden_number_begin):
            A_hidden_begin.put(k,logsig(float(A_hidden_begin.take(k).getA())))
        A_output_begin=locals()['W_output_'+j]*A_hidden_begin+locals()['B_output_'+j]
        for k in range(output_number_begin):
            A_output_begin.put(k,logsig(float(A_output_begin.take(k).getA())))
        matrix_list.append(float(A_output_begin.take(0).getA()))
    P_input_final=numpy.mat(matrix_list).T
    A_hidden_final=W_hidden_final*P_input_final+B_hidden_final
    for j in range(hidden_number_final):
        A_hidden_final.put(j,logsig(float(A_hidden_final.take(j).getA())))
    A_output_final=W_output_final*A_hidden_final+B_output_final
    for j in range(output_number_final):
        A_output_final.put(j,logsig(float(A_output_final.take(j).getA())))
    fout_predict.write(str(float(A_output_final.take(j).getA()))+'	'+'F'+'\n')
    if float(A_output_final.take(0).getA()) < 0.5:
        nn_false=nn_false+1

fout_correct.write(str(float((np_true+nn_false)(np_total+nn_total))))

fout_predict.close()
fout_correct.close()
