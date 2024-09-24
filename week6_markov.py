import numpy as np
import itertools
import pandas as pd

# create state space and initial state probabilities
states = ['sleeping','eating','pooping']

hidden_states = ['healthy','sick']
pi = [0.5,0.5]
state_space = pd.Series(pi,index=hidden_states,name='states')
print(state_space)

a_df = pd.DataFrame(columns=hidden_states,index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7,0.3]
a_df.loc[hidden_states[1]] = [0.4,0.6]

print(a_df)

observable_states = states

b_df = pd.DataFrame(columns=observable_states,index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2,0.6,0.2]
b_df.loc[hidden_states[1]] = [0.4,0.1,0.5]

print(b_df)

def HMM(obsq,a_df,b_df,pi,states,hidden_states):
    hidst = list(itertools.product(hidden_states,repeat=len(obsq)))
    print(hidst)
    sum=0
    for k in hidst:
        prod=1
        for j in range(len(k)):
            c=0
            for i in obsq:
                if c==0:
                    prod*=a_df[i][k[j]]*pi[hidden_states.index(k[j])]
                    c=1
                else:
                    prod*=b_df[k[j]][k[j-1]]*a_df[i][k[j]]
        sum +=prod
        c=0
    return sum

def vertibi(obsq,a_df,b_df,pi,states,hidden_states):
    sum=0
    hidst = list(itertools.product(hidden_states, repeat=len(obsq)))
    for k in hidst:
        sum1=0
        prod=1
        for j in range(len(k)):
            c=0
            for i in obsq:
                if c==0:
                    prod *= a_df[i][k[j]] * pi[hidden_states.index(k[j])]
                    c = 1
                else:
                    prod*=b_df[k[j]][k[j-1]]*a_df[i][k[j]]
        c=0
        sum1+=prod
        if(sum1>sum):
            sum=sum1
            hs=k
    return sum,hs
obsq=['eating','sleeping','sleeping']
print(HMM(obsq,b_df,a_df,pi,states,hidden_states))
print(vertibi(obsq,b_df,a_df,pi,states,hidden_states))
