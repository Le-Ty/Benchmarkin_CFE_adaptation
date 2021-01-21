import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('white')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

## for Palatino and other serif fonts use:
font = {'family' : 'serif',
        'weight' : 'bold',
        'serif':['Palatino'],
        'size'   : 15}
rc('font',**font)
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def main():

    classifier_name = 'ANN'
    data_name = 'Adult'
    grouped = True
    reading_names = ['ar', 'as', 'cem', 'clue', 'dice', 'dice_vae', 'face', 'gs']
    names = ['ar-lime', 'as', 'cem', 'clue', 'dice', 'dice-vae', 'face', 'gs']


    results = pd.read_csv('Results/{}/{}/{}.csv'.format(data_name, classifier_name, reading_names[0]))
    results['Unnamed: 0'] = np.repeat(names[0], results.values.shape[0])

    for i in range(1, len(names)):
        to_add = pd.read_csv('Results/Adult/{}/{}.csv'.format(classifier_name, reading_names[i]))
        to_add['Unnamed: 0'] = np.repeat(names[i], to_add.values.shape[0])
        results = results.append(to_add)

    results = results.rename(columns={'Unnamed: 0': 'Methods'})


    # Plotting begins
    names_interest = ['ell0', 'redundancy', 'ell1', 'ell2']
    
    names_interest1 = ['ell0', 'redundancy']
    x_label_names1 = [r'$||\delta_x||_0$', 'redundancy']
    
    names_interest2 = ['ell1', 'ell2']
    x_label_names2 = [r'$||\delta_x||_1$', r'$||\delta_x||_2$']

    # Basic violinplot options
    # optional: scale options: {“area”, “count”, “width”}
    # optional: inner{“box”, “quartile”, “point”, “stick”, None}
    
    if grouped:
        
        group = 'group_true'
        # First round of plots
        fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
        dfg = results.groupby('Methods')
        
        sns.violinplot(x=names_interest1[0], y="Methods", data=results, palette="Blues",
                       cut=0, scale='count', inner='quartile', ax=axs[0])
        axs[0].set_ylabel('Methods')
        axs[0].set_xlabel(r'$||\delta_x||_0$')
        axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot


        sns.violinplot(x=names_interest1[1], y="Methods", data=results, palette="Blues",
                       cut=0, scale='count', inner='quartile', ax=axs[1])
        axs[1].set_xlabel('redundancy')
        axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
        # Remove second y label
        axs[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, classifier_name, data_name, classifier_name,
                                                              group, names_interest1[0], names_interest1[1]))
        plt.show()

        # Second round of plots
        fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
        dfg = results.groupby('Methods')

        sns.violinplot(x=names_interest1[0], y="Methods", data=results, palette="Blues",
                       cut=0, scale='count', inner='quartile', ax=axs[0])
        axs[0].set_xlabel(r'$||\delta_x||_1$')
        axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot

        sns.violinplot(x=names_interest1[1], y="Methods", data=results, palette="Blues",
                       cut=0, scale='count', inner='quartile', ax=axs[1])
        axs[1].set_xlabel(r'$||\delta_x||_2$')
        axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
        axs[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, classifier_name, data_name, classifier_name,
                                                              group, names_interest2[0], names_interest2[1]))
        plt.show()

    ## TODO: Add support for plotting each plot one by one
    '''
    else:
        group = 'group_false'
        ax = sns.violinplot(x="ell0", y="Methods", data=results, palette="Blues",
                                cut=0, scale='count', inner='quartile')
    
        dfg = results.groupby('Methods')
    
        ax.set_ylabel('Methods')
        ax.set_xlabel(r'$||\delta_x||_0$')
        ax.set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
        plt.tight_layout()
        plt.show()
    '''
    
if __name__ == "__main__":
    # execute only if run as a script
    main()