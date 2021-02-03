import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('white')
import matplotlib.pyplot as plt
from matplotlib import rc

## for Palatino and other serif fonts use:
font = {'family' : 'serif',
        'weight' : 'bold',
        'serif':['Palatino'],
        'size': 12}
rc('font',**font)
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def main():

    classifier_name = 'ANN'
    data_name = 'Compas'
    grouped = True

    # if data_name == 'X':
    #     reading_names = ['clue', 'face-eps', 'face-knn', 'dicfe']
    #     names = ['clue', 'face-eps', 'face-knn', 'dicfe']
    #
    #     reading_names_dependence = ['clue', 'face-eps', 'face-knn', 'dicfe']
        #
        # names_dependence =['clue', 'face-eps', 'face-knn', 'dicfe']
    # else:
    reading_names = ['ar', 'as', 'cem', 'cem-vae', 'dice', 'face-eps', 'face-knn', 'gs', 'dicfe', 'dicfe-dir', 'dice-indir'] #, 'dicfe_indep']
    names = ['ar-lime', 'as', 'cem', 'dice',  'face', 'gs', 'dicfe', 'dicfe-dir', 'dice-indir'] #, 'indep']

    reading_names_dependence = [ 'face-knn', 'face-eps', 'dicfe', 'dicfe-dir', 'dicfe-indir'] #, 'dicfe_indep']

    names_dependence = [ 'face-knn', 'face-eps', 'dicfe','dicfe-dir', 'dicfe-indir'] #, 'indep']


    # Dependence Results to DF
    results_dependence = pd.read_csv('Results/{}/{}/{}.csv'.format(data_name, classifier_name,
                                                                   reading_names_dependence[0]))
    results_dependence['Unnamed: 0'] = np.repeat(names_dependence[0],
                                                 results_dependence.values.shape[0])
    for i in range(1, len(names_dependence)):
        to_add2 = pd.read_csv('Results/Compas/{}/{}.csv'.format(classifier_name,
                                                              reading_names_dependence[i]))
        to_add2['Unnamed: 0'] = np.repeat(names_dependence[i],
                                         to_add2.values.shape[0])
        results_dependence = results_dependence.append(to_add2)
    results_dependence = results_dependence.rename(columns={'Unnamed: 0': 'Methods'})

    # Collect results & introduce 'Assumption' category
    results_dependence['Assumption'] = np.repeat('Dependent', results_dependence.values.shape[0])
    results = results_dependence

    # Print latex code to console
    results_table = results[['Methods', 'Assumption', 'ynn', 'success', 'avgtime']].dropna()
    print(results_table.to_latex())

    # Plotting begins
    names_interest1 = ['ell0', 'redundancy']
    names_interest2 = ['ell1', 'ell2']

    # Basic violinplot options
    # optional: scale options: {“area”, “count”, “width”}
    # optional: inner{“box”, “quartile”, “point”, “stick”, None}

    if grouped:

        group = 'group_true'
        # First round of plots
        fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
        dfg = results.groupby('Methods')

        sns.violinplot(x=names_interest1[0], y="Methods", hue="Assumption", data=results, palette="Blues",
                       cut=0, scale='count', inner='box', ax=axs[0], legend_out=True)
        #axs[0].set_ylabel('Methods')
        axs[0].set_xlabel(r'$||\delta_x||_0$')
        #axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot


        sns.violinplot(x=names_interest1[1], y="Methods", hue="Assumption", data=results, palette="Blues",
                       cut=0, scale='count', inner='box', ax=axs[1], legend_out=False)
        axs[1].set_xlabel('redundancy')
        #axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
        # Remove second y label
        axs[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, classifier_name, data_name, classifier_name,
                                                              group, names_interest1[0], names_interest1[1]))
        plt.show()

        # Second round of plots
        fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
        dfg = results.groupby('Methods')

        sns.violinplot(x=names_interest2[0], y="Methods", hue="Assumption", data=results, palette="Blues",
                       cut=0, scale='count', inner='box', ax=axs[0])
        axs[0].set_xlabel(r'$||\delta_x||_1$')
        #axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot

        sns.violinplot(x=names_interest2[1], y="Methods", hue="Assumption", data=results, palette="Blues",
                       cut=0, scale='count', inner='box', ax=axs[1])
        axs[1].set_xlabel(r'$||\delta_x||_2$')
        #axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
        axs[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, classifier_name, data_name, classifier_name,
                                                              group, names_interest2[0], names_interest2[1]))
        plt.show()

    ## TODO: Add support for plotting each plot one by one

    else:
        group = 'group_false'
        names_interest = ['ell0', 'redundancy', 'ell1', 'ell2']
        x_label_names = [r'$||\delta_x||_0$', 'redundancy', r'$||\delta_x||_1$', r'$||\delta_x||_2$']

        for i in range(len(names_interest)):

            fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
            sns.violinplot(x=names_interest[i], y="Methods", hue="Assumption", data=results, palette="Blues",
                                    cut=0, scale='count', inner='quartile')
            dfg = results.groupby('Methods')

            ax.set_ylabel('Methods')
            ax.set_xlabel(x_label_names[i])
            #ax.set_yticklabels(['%s\n$n=$%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot

            #plt.tight_layout()
            fig.savefig('Results/{}/{}/Plots/{}_{}_{}_{}.pdf'.format(data_name, classifier_name, data_name, classifier_name,
                                                                  group, names_interest[i]))

            plt.show()
            ax.clear()
            fig.clear()


if __name__ == "__main__":
    # execute only if run as a script
    main()
