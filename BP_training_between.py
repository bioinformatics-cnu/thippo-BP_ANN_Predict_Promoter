import numpy

promoter_filename='train_Tset_m1.th'
not_promoter_filename='train_Fset_m1.th'
seq_sorting_list=['ATCG', 'ATGC', 'ACTG', 'ACGT', 'AGTC', 'AGCT', 'TACG', 'TAGC', 'TCAG', 'TCGA', 'TGAC', 'TGCA', 'CATG', 'CAGT', 'CTAG', 'CTGA', 'CGAT', 'CGTA', 'GATC', 'GACT', 'GTAC', 'GTCA', 'GCAT', 'GCTA']
input_number_begin=324
hidden_number_begin=81
output_number_begin=1

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

fout_promoter_next=open('promoter_next.th','w')
for i in promoter_list:
    for j in seq_sorting_list:
        P_input_begin=numpy.mat(seq_translate_list(j,i)).T
        A_hidden_begin=locals()['W_hidden_'+j]*P_input_begin+locals()['B_hidden_'+j]
        for k in range(hidden_number_begin):
            A_hidden_begin.put(k,logsig(float(A_hidden_begin.take(k).getA())))
        A_output_begin=locals()['W_output_'+j]*A_hidden_begin+locals()['B_output_'+j]
        for k in range(output_number_begin):
            A_output_begin.put(k,logsig(float(A_output_begin.take(k).getA())))
        fout_promoter_next.write(str(float(A_output_begin.take(0).getA()))+';')
    fout_promoter_next.write('\n')
fout_promoter_next.close()

fout_not_promoter_next=open('not_promoter_next.th','w')
for i in not_promoter_list:
    for j in seq_sorting_list:
        P_input_begin=numpy.mat(seq_translate_list(j,i)).T
        A_hidden_begin=locals()['W_hidden_'+j]*P_input_begin+locals()['B_hidden_'+j]
        for k in range(hidden_number_begin):
            A_hidden_begin.put(k,logsig(float(A_hidden_begin.take(k).getA())))
        A_output_begin=locals()['W_output_'+j]*A_hidden_begin+locals()['B_output_'+j]
        for k in range(output_number_begin):
            A_output_begin.put(k,logsig(float(A_output_begin.take(k).getA())))
        fout_not_promoter_next.write(str(float(A_output_begin.take(0).getA()))+';')
    fout_not_promoter_next.write('\n')
fout_not_promoter_next.close()
