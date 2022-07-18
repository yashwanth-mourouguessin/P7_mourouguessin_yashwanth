import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_bar(df, SK_ID_CURR, col):
    df_per_personne = df.loc[df.SK_ID_CURR == SK_ID_CURR]

    df_f_1 = pd.DataFrame(df[col].value_counts())
    index = list(df_f_1.index)
    values = list(df_f_1[col])
    fig = plt.figure(figsize = (30, 5))
    v_1 = df_per_personne[col].loc[0]
    
    clrs = ['red' if x==v_1 else 'grey' for x in index]
    plt.bar(index, values, width=0.1, color=clrs)
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('repartition')
    plt.title("repartition des données selon la variable "+col)
    plt.show()
    
def hit_plot(df, SK_ID_CURR, col):
    df_per_personne = df.loc[df.SK_ID_CURR == SK_ID_CURR]
    v_1 = df_per_personne[col].iloc[0]
    
    if col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
               'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_CONT_MOBILE']:
        x = np.log(df[col])
        n, bins, patches = plt.hist(x.replace(-np.inf, -10000000000), bins=10)
        v_1 = np.log(v_1)
        if v_1==-np.inf:
            v_1 = -10000000000
        for i in range(len(bins)-1):
            if bins[i]<=v_1 and v_1<=bins[i+1]:
                patches[i].set_facecolor('red')
            else:
                patches[i].set_facecolor('grey')
        plt.xlabel(col)
        plt.ylabel('repartition')
        plt.title("repartition des données selon la variable "+col)
        plt.show()

    else:
        n, bins, patches = plt.hist(df[col], bins=10)
        for i in range(len(bins)-1):
            if bins[i]<=v_1 and v_1<bins[i+1]:
                patches[i].set_facecolor('red')
            else:
                patches[i].set_facecolor('grey')
        if bins[i]<v_1 and v_1<=bins[i+1]:
            patches[i].set_facecolor('red')
        else:
            patches[i].set_facecolor('grey')
        plt.xlabel(col)
        plt.ylabel('repartition')
        plt.title("repartition des données selon la variable "+col)
        plt.show()
    

def final(df, SK_ID_CURR, col):
    if df[col].dtype==int or df[col].dtype==float:
        hit_plot(df, SK_ID_CURR, col)
    else:
        plot_bar(df, SK_ID_CURR, col)
