import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('white')
import matplotlib.pyplot as plt
from matplotlib import rc
import library.measure as measure

# ## for Palatino and other serif fonts use:
# font = {'family' : 'serif',
#         'weight' : 'bold',
#         'serif':['Palatino'],
#         'size': 12}
# rc('font',**font)
# rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
#




def main():

    model_name = 'ANN'
    # file_name = 'distances_raw'
    data_name = 'Adult'

    #preprocessing
    reading_names = ['sum_per_row_f_rec', 'sum_per_row_cf', 'sum_per_row_f_orig']

    fact_rec = pd.read_csv('Results/{}/{}/fact_rec.csv'.format(data_name, model_name), index_col = False)
    cfs = pd.read_csv('Results/{}/{}/CF.csv'.format(data_name, model_name), index_col = False)
    data =  pd.read_csv("CF_Examples/counterfact_expl/CE/Datasets/data_processed/" + data_name + "/X_train.csv", index_col = False)
    fact_orig = pd.read_csv("CF_Examples/counterfact_expl/CE/Datasets/data_processed/" + data_name + "/X_query.csv", index_col = False)
    print(fact_rec.head(5))
    print(cfs.head(5))
    print(fact_orig.head(5))


    range = measure.get_range(data)
    delta_dir = (cfs.iloc[:,1:] - fact_rec.iloc[:,1:]).values
    delta_indir = (fact_orig.iloc[:,1:].head(50) - fact_rec.iloc[:,1:]).values
    delta_total = (cfs.iloc[:,1:] - fact_orig.iloc[:,1:].head(50)).values
    print(range)
    print(delta_dir[0])

    l1_total = np.abs(delta_total)
    l1_total = np.sum(l1_total, axis = 1)

    l1_dir = np.abs(delta_dir) #[np.abs(x[0] / x[1]) for x in zip(delta_dir, range)]
    print(l1_dir)
    l1_dir = np.sum(l1_dir, axis = 1)
    print(l1_dir)
    # print(delta_indir)

    l1_indir = np.abs(delta_indir) #[np.abs(x[0] / x[1]) for x in zip(delta_indir, range)]
    l1_indir = np.sum(l1_indir, axis = 1)

    l2_total = delta_total**2
    l2_total = np.sum(l2_total, axis = 1)


    l2_dir = delta_dir**2 #[(x[0] / x[1])**2 for x in zip(delta_dir, range)]
    # print(l2_dir[:10])
    l2_dir = np.sum(l2_dir, axis = 1)

    l2_indir = delta_indir**2 #[(x[0] / x[1])**2 for x in zip(delta_indir, range)]
    l2_indir = np.sum(l2_indir, axis = 1)

    # print(l1_indir)
    # print(l2_dir)
    # print(l2_indir)

    distance = pd.DataFrame(l1_dir, columns = ['l1-dir'])
    distance = distance.assign(l1indir =  l1_indir)
    distance = distance.assign(l2dir =  l2_dir)
    distance = distance.assign(l2indir =  l2_indir)
    distance = distance.assign(totall1 = l1_total)
    distance = distance.assign(totall2 = l2_total)

    dist_melt = distance.melt(var_name = "feature", value_name = "values")
    # sns.set(font_scale=2)  # crazy big

    d = sns.violinplot(data = dist_melt, x = "values", y = "feature", scale = "width", palette = "Blues", inner = 'box', cut = 0).set_title(" Direct vs. Indirect Cost")

    d_plot = d.get_figure()

    d_plot.savefig('DirIndir_error.pdf')

    distance.to_csv('DirIndir_distance.csv')
    print(distance)






    #
    # names = ['l1 direct', 'l1 indirect', 'l2 direct', 'l2 indirect']
    #
    # # Dependence Results to DF
    # results['Unnamed: 0'] = np.repeat(names[0], results.values.shape[0])
    # for i in range(1, len(names)):
    #     to_add2 = pd.read_csv('Results/Adult/{}/{}.csv'.format(file_name,
    #                                                           reading_names[i]))
    #     to_add2['Unnamed: 0'] = np.repeat(names[i],
    #                                      to_add2.values.shape[0])
    #     results = results.append(to_add2)
    # results = results.rename(columns={'Unnamed: 0': 'Methods'})
    #
    # # Collect results & introduce 'Assumption' category
    # # results['Assumption'] = np.repeat('Dependent', results_dependence.values.shape[0])
    #
    # # Print latex code to console
    # results_table = results[['l1 direct', 'l1 indirect', 'l2 direct', 'l2 indirect']].dropna()
    # print(results_table.to_latex())
    #
    # # Plotting begins
    # names_interest1 = ['ell1 direct', 'ell2 direct']
    # names_interest2 = ['ell1 indirect', 'ell2 indirect']
    #
    #
    #
    # if grouped:
    #
    #     group = 'group_true'
    #     # First round of plots
    #     fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
    #     dfg = results.groupby('Methods')
    #
    #     sns.violinplot(x=names_interest1[0], y="Methods", data=results, palette="Blues",
    #                    cut=0, scale='count', inner='box', ax=axs[0], legend_out=True)
    #     #axs[0].set_ylabel('Methods')
    #     axs[0].set_xlabel(r'$||\delta_x||_1$ direct')
    #     #axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot
    #
    #
    #     sns.violinplot(x=names_interest1[1], y="Methods", data=results, palette="Blues",
    #                    cut=0, scale='count', inner='box', ax=axs[1], legend_out=False)
    #     axs[1].set_xlabel('$||\delta_x||_1$ indirect')
    #     #axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
    #     # Remove second y label
    #     axs[1].set_ylabel('')
    #
    #     plt.tight_layout()
    #     plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, file_name, data_name, file_name,
    #                                                           group, names_interest1[0], names_interest1[1]))
    #     plt.show()
    #
    #     # Second round of plots
    #     fig, axs = plt.subplots(nrows=1, ncols=2, num=1)
    #     dfg = results.groupby('Methods')
    #
    #     sns.violinplot(x=names_interest2[0], y="Methods", data=results, palette="Blues",
    #                    cut=0, scale='count', inner='box', ax=axs[0])
    #     axs[0].set_xlabel(r'$||\delta_x||_2$ direct')
    #     #axs[0].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements 'n' to plot
    #
    #     sns.violinplot(x=names_interest2[1], y="Methods", data=results, palette="Blues",
    #                    cut=0, scale='count', inner='box', ax=axs[1])
    #     axs[1].set_xlabel(r'$||\delta_x||_2$ indirect')
    #     #axs[1].set_yticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])  # add number elements to plot
    #     axs[1].set_ylabel('')
    #
    #     plt.tight_layout()
    #     plt.savefig('Results/{}/{}/Plots/{}_{}_{}_{}_{}.pdf'.format(data_name, file_name, data_name, file_name,
    #                                                           group, names_interest2[0], names_interest2[1]))
    #     plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main()
