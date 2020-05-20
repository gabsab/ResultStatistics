# ********************************************************************************
# Evaluate data of buckling for HSS sections

# Created for the Aeolus4future project
# Author: Gabriel Sabau
# ********************************************************************************
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Polygon
import math
import seaborn as sns
import pandas as pd




def integrate_area(coord):
    coord2 = coord[1:]
    coord1 = coord[:-1]
    A = 0
    for i1, i2 in zip(coord1, coord2):
        (x1, y1) = i1
        (x2, y2) = i2
        A = A + (x1+x2)*(y2-y1)/2
    return A


File_name = 'C:\\Users\\gabsab\\Documents\\HstrSteel\\Excel_files\\PresentData2.xlsx'
File_name = 'H:\\Documents\\HstrSteel\\Excel_files\\PresentData2.xlsx'
Sheet_name = 'Curve_c'    # sheet name 'All'

dbase = pd.read_excel(File_name, Sheet_name, header=0)

Sheet_name = 'Curve_b'    # sheet name 'All'

dbase_b = pd.read_excel(File_name, Sheet_name, header=0)


########################################################################################################################
#pyplot


def regression_models(database, name, save_dir):
    inch = 1
    cm = inch / 2.54
    matplotlib.style.use('default')
    fig, ax = plt.subplots(nrows=1, ncols=2,  figsize=[17*cm, 5*cm], sharey='row', constrained_layout=True,
                           gridspec_kw={"wspace": 0.05, 'width_ratios': [5, 1]})

    sns.set_context("paper")

    sns.set_style("whitegrid",
                  rc={'axes.grid': True, 'grid.linestyle': '-', 'grid.color': 'xkcd:light grey', 'xtick.bottom': True,
                      'ytick.left': True, 'marker.size': '0.5', "xtick.direction": "in", "ytick.direction": "in",
                      'font.size': 10, 'font': 'calibri'})
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    ax[0].set_xticks(np.arange(0, 2.6, 0.2))
    ax[0].set_xlim([0, 2.401])
    ax[0].set_ylim([0.7, 1.9])
    ax[0].grid(which='major', axis='both', linewidth=0.5, zorder=1)
    ax[0].plot([0, 2.5], [1, 1], linestyle='-', color='grey', linewidth=2, zorder=2)
    regression = (sns.regplot(x_ci=90, x="Slenderness", y=name, x_bins=np.arange(0, 2.4, 0.2), x_estimator=np.mean,
                              data=database, ax=ax[0], color='black', fit_reg=True, truncate=True, marker="_",
                              line_kws={'linewidth': '1.0'}))
    # sns.scatterplot(x="Slenderness", y=name, hue="Steel Grade", data=database, hue_order={'S960', 'S700', 'S690'},
    #                 **{"markers": ["s", "^", "o"], "color": "black"},
    #                 ax=ax[0])


    plt.rc('font', family='calibri')
    ax[0].set_xlabel(r'Non-dimensional slenderness, ${\overline{\lambda}}$')
    ax[0].set_ylabel('$N_{exp}/{N_{Rk}}$')

    ax[1].set_ylim([0.7, 1.9])

    distribution = (sns.distplot(database[name].dropna(), ax=ax[1], axlabel=False, vertical=True, rug=False,
                                 color='black',
                                 bins=np.arange(0.6, math.ceil(np.max(database[name].dropna())*10)/10, 0.1),
                                 kde=True, kde_kws={"linestyle": "--", "linewidth": 0.5, "zorder": 3, "bw": 'scott'}))
                                 #fit=stats.norm, fit_kws={"linewidth": 1.0}))

    res_mean, res_std = stats.norm.fit(database[name].dropna())
    lnspc = np.linspace(res_mean-4*res_std, res_mean+4*res_std, 20*len(database[name].dropna()))
    fit_pdf = stats.norm.pdf(lnspc, res_mean, res_std)
    ax[1].plot(fit_pdf, lnspc, lw=1, color='black', label='Norm PDF', zorder=4)
    ix = []
    iy = []
    for j, i in zip(lnspc, fit_pdf):
        if j >= 1-0.0025:
                iy.append(j)
                ix.append(i)
    verts= [(0.0, iy[0]), *zip(ix, iy), (0.0, iy[-1])]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.5', alpha=0.6)
    ax[1].add_patch(poly)
    Probability = integrate_area(verts)*100
    chart_text = "%.1f %%" % Probability
    plt.text(0.1, res_mean, chart_text, rotation=270, verticalalignment='center', fontsize=10, zorder=10)
    lower_90 = res_mean-1.64*res_std
    ci_text = "%.3f " % lower_90

    xmin, xmax = ax[1].get_xlim()
    ax[1].set_xlim([xmin, math.ceil(xmax)+0.01])

    if lower_90 >= 0.75:
        plt.text(xmax, lower_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                 fontsize=10, rotation='horizontal')
    higher_90 = res_mean + 1.64 * res_std
    ci_text = "%.3f " % higher_90
    if higher_90 <= np.max(database[name]):
        plt.text(xmax, higher_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                 fontsize=10, rotation='horizontal', zorder=10)
    ax[1].plot([xmin, xmax+1], [lower_90, lower_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [higher_90, higher_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [1, 1], '-', color='grey', linewidth=2, zorder=2)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[1].set_xticks([]))
    #plt.xlabel('PDF ', fontsize=10)
    #ax[1].xlabel(None)

    quant2 = np.quantile(database[name].dropna(), 0.05, interpolation='nearest')
    cov = stats.variation(database[name].dropna())

    info_text = r'$\overline{x}$ = ' + str(np.round(res_mean, 3)) + '\n' + r"$s$ = " + str(np.round(res_std, 3)) \
                + '\n' + r'$COV$ = ' + str(np.round(cov, 3)) # + '\n' + r'$5\%fractile$ = ' + str(np.round(quant2, 3))

    h = ax[0].text(0.0120, 0.97, str(info_text), color='black', horizontalalignment='left',
                   verticalalignment='top', fontsize=10, rotation='horizontal', zorder=10, transform=ax[0].transAxes,
                   bbox={'facecolor': 'white', 'edgecolor': 'black'})  # 'pad':10
    fig_title = save_dir + name+".png"
    plt.savefig(fig_title,  format='png', dpi=1000) #bbox_inches='tight',
    fig_title = save_dir + name + ".pdf"
    plt.savefig(fig_title, format='pdf', dpi=1000) #bbox_inches='tight'
    #plt.show()
    plt.close(fig='all')

    file_name = save_dir + name + ".txt"
    stat_file = open(file_name, "w")
    print("Model compared: \t", name, "\n###################################################",
          "\nNumber of specimens: \t", len(database[name].dropna()),
          "\nResiduals Statistics (deviation from proposed model):",
          "\nMean value: \t", res_mean, "\nStandard error: \t", stats.sem(database[name].dropna()),
          "\nStandard deviation: \t", res_std, "\n95% Confidence interval \tlower: \t", lower_90,
          "\tupper: \t", higher_90, "\nProbability of sample being above the limit:", Probability,
          "\n5% lower quantile: \t", quant2, "\nCOV: \t", cov, file=stat_file)
    stat_file.close()


def regression_models2(database, name, fabrication, save_dir):
    inch = 1
    cm = inch / 2.54
    database = database[database['Fabrication'] == fabrication]
    matplotlib.style.use('default')
    #fig_kw = {"constrained_layout.h_pad":  0.1 * cm, "constrained_layout.w_pad":  0.1 * cm}
    fig, ax = plt.subplots(nrows=1, ncols=2,  figsize=[17*cm, 5*cm], sharey='row', constrained_layout=True,
                           gridspec_kw={"wspace": 0.05, 'width_ratios': [5, 1]}, )
    sns.set_context("paper")

    sns.set_style("whitegrid",
                  rc={'axes.grid': True, 'grid.linestyle': '-', 'grid.color': 'xkcd:light grey', 'xtick.bottom': True,
                      'ytick.left': True, 'marker.size': '0.5', "xtick.direction": "in", "ytick.direction": "in",
                      'font.size': 10, 'font': 'calibri'})
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    ax[0].set_xticks(np.arange(0, 2.6, 0.2))
    xmin0, xmax0 = [0.0, 2.401]
    ymin0, ymax0 = [0.7, 1.9]
    ax[0].set_xlim([xmin0, xmax0])
    ax[0].set_ylim([ymin0, ymax0])
    ax[0].grid(which='major', axis='both', linewidth=0.5, zorder=1)
    ax[0].plot([0, 2.5], [1, 1], linestyle='-', color='grey', linewidth=2, zorder=2)
    matplotlib.rcParams['errorbar.capsize'] = 20
    regression = (sns.regplot(x_ci=90, x="Slenderness", y=name, #x_bins=np.arange(0, 2.4, 0.2), x_estimator=np.mean,
                              data=database, ax=ax[0], color='black', fit_reg=True, truncate=True, marker="",
                              line_kws={'linewidth': '1.0'}))

    s = database.groupby("Steel Grade").size()
    grades = s.index[s > 0]

    for steel_grade in grades:
        cond_markers = ['s' if steel_grade == 'S960' else '^' if steel_grade == 'S700' else 'o']
        sc = ax[0].scatter(x="Slenderness", y=name, data=database[database['Steel Grade'] == steel_grade],
                           marker=cond_markers[0], label=steel_grade, c='black', zorder=8)
        if cond_markers == ['o']:
            sc.set(facecolors='none', edgecolors='black')

    label_list = ax[0].get_legend_handles_labels()[1]
    label_list.reverse()
    handles_list = ax[0].get_legend_handles_labels()[0]
    handles_list.reverse()
    ax[0].legend(handles_list, label_list, loc=1)


    # sns.scatterplot(x="Slenderness", y=name, hue="Steel Grade", data=database, hue_order={'S960', 'S700', 'S690'},
    #                 **{"markers": ["s", "^", "o"], "color": "black"},
    #                 ax=ax[0])
    plt.rc('font', family='calibri')
    ax[0].set_xlabel(r'Non-dimensional slenderness, ${\overline{\lambda}}$')
    ax[0].set_ylabel('$N_{exp}/{N_{Rk}}$')

    ax[1].set_ylim([0.7, 1.9])

    distribution = (sns.distplot(database[name].dropna(), ax=ax[1], axlabel=False, vertical=True, rug=False,
                                 bins=np.arange(0.6, math.ceil(np.max(database[name].dropna())*10)/10, 0.1),
                                 color='black',
                                 kde=True, kde_kws={"linestyle": "--", "linewidth": 0.5, "zorder": 3, "bw": 'scott'}))
                                 #fit=stats.norm, fit_kws={"linewidth": 1.0}))

    res_mean, res_std = stats.norm.fit(database[name].dropna())
    lnspc = np.linspace(res_mean-4*res_std, res_mean+4*res_std, 20*len(database[name].dropna()))
    fit_pdf = stats.norm.pdf(lnspc, res_mean, res_std)
    ax[1].plot(fit_pdf, lnspc, lw=1, color='black', label='Norm PDF', zorder=4)
    ix = []
    iy = []
    for j, i in zip(lnspc, fit_pdf):
        if j >= 1-0.0025:
                iy.append(j)
                ix.append(i)
    verts= [(0.0, iy[0]), *zip(ix, iy), (0.0, iy[-1])]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.5', alpha=0.6)
    ax[1].add_patch(poly)
    Probability = integrate_area(verts)*100
    chart_text = "%.1f %%" % Probability
    plt.text(0.1, res_mean, chart_text, rotation=270, verticalalignment='center', fontsize=10, zorder=10)
    lower_90 = res_mean-1.64*res_std
    ci_text = "%.3f " % lower_90

    xmin, xmax = ax[1].get_xlim()
    ax[1].set_xlim([xmin, math.ceil(xmax)+0.01])

    if lower_90 >= 0.75:
        plt.text(xmax, lower_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                 fontsize=10, rotation='horizontal')
    higher_90 = res_mean + 1.64 * res_std
    ci_text = "%.3f " % higher_90
    if higher_90 <= np.max(database[name]):
        plt.text(xmax, higher_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                 fontsize=10, rotation='horizontal', zorder=10)
    ax[1].plot([xmin, xmax+1], [lower_90, lower_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [higher_90, higher_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [1, 1], '-', color='grey', linewidth=2, zorder=2)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[1].set_xticks([]))
    #plt.xlabel('PDF ', fontsize=10)
    #ax[1].xlabel(None)

    quant2 = np.quantile(database[name].dropna(), 0.05, interpolation='nearest')
    cov = stats.variation(database[name].dropna())
    info_text = r'$\overline{x}$ = ' + str(np.round(res_mean, 3)) + '\n' + r"$s$ = " + str(np.round(res_std, 3)) \
                + '\n' + r'$COV$ = ' + str(np.round(cov, 3))  # + '\n' + r'$5\%fractile$ = ' + str(np.round(quant2, 3))

    h = ax[0].text(0.0120, 0.97, str(info_text), color='black', horizontalalignment='left',
                   verticalalignment='top', fontsize=10, rotation='horizontal', zorder=10, transform=ax[0].transAxes,
                   bbox={'facecolor': 'white', 'edgecolor': 'black'}) # 'pad':10
    fig_title = save_dir + name + fabrication + ".png"
    plt.savefig(fig_title, format='png', dpi=1000) #bbox_inches='tight'
    fig_title = save_dir + name + fabrication + ".pdf"
    plt.savefig(fig_title, format='pdf', dpi=1000) #bbox_inches='tight'
    fig_title = save_dir + name + fabrication + ".jpg"
    plt.savefig(fig_title, format='jpg', dpi=1000)  # bbox_inches='tight'
    #plt.show()
    plt.close(fig='all')
    file_name = save_dir + name + fabrication + ".txt"
    stat_file = open(file_name, "w")
    print("Model compared: \t", name, "\n###################################################",
          "\nNumber of specimens: \t", len(database[name].dropna()),
          "\nResiduals Statistics (deviation from proposed model):",
          "\nMean value: \t", res_mean, "\nStandard error: \t", stats.sem(database[name].dropna()),
          "\nStandard deviation: \t", res_std, "\n95% Confidence interval \tlower: \t", lower_90,
          "\tupper: \t", higher_90, "\nProbability of sample being above the limit:", Probability,
          "\n5% lower quantile: \t", quant2, "\nCOV: \t", cov, file=stat_file)
    stat_file.close()


# def regression_models_lognormal(database, name, fabrication):
#     inch = 1
#     cm = inch / 2.54
#     database = database[database['Fabrication'] == fabrication]
#     matplotlib.style.use('default')
#     #fig_kw = {"constrained_layout.h_pad":  0.1 * cm, "constrained_layout.w_pad":  0.1 * cm}
#     fig, ax = plt.subplots(nrows=1, ncols=2,  figsize=[17*cm, 6*cm], sharey='row', constrained_layout=True,
#                            gridspec_kw={"wspace": 0.05, 'width_ratios': [5, 1]}, )
#     sns.set_context("paper")
#
#     sns.set_style("whitegrid",
#                   rc={'axes.grid': True, 'grid.linestyle': '-', 'grid.color': 'xkcd:light grey', 'xtick.bottom': True,
#                       'ytick.left': True, 'marker.size': '0.5', "xtick.direction": "in", "ytick.direction": "in",
#                       'font.size': 10, 'font': 'calibri'})
#     ax[0].set_xticks(np.arange(0, 2.6, 0.2))
#     xmin0, xmax0 = [0.0, 2.401]
#     ymin0, ymax0 = [0.7, 1.9]
#     ax[0].set_xlim([xmin0, xmax0])
#     ax[0].set_ylim([ymin0, ymax0])
#     ax[0].grid(which='major', axis='both', linewidth=0.5, zorder=1)
#     ax[0].plot([0, 2.5], [1, 1], linestyle='-', color='grey', linewidth=2, zorder=2)
#     matplotlib.rcParams['errorbar.capsize'] = 20
#     regression = (sns.regplot(x_ci=90, x="Slenderness", y=name, x_bins=np.arange(0, 2.4, 0.2), x_estimator=np.mean,
#                               data=database, ax=ax[0], color='black', fit_reg=True, truncate=False, marker="_",
#                               line_kws={'linewidth': '1.0'}))
#     # sns.scatterplot(x="Slenderness", y=name, hue="Steel Grade", data=database, hue_order={'S960', 'S700', 'S690'},
#     #                 **{"markers": ["s", "^", "o"], "color": "black"},
#     #                 ax=ax[0])
#     plt.rc('font', family='calibri')
#     ax[0].set_xlabel(r'Non-dimensional slenderness, ' r'${\lambda}$')
#     ax[0].set_ylabel(r'${N_u}/{N_{Rk}}$')
#
#     ax[1].set_ylim([0.7, 1.9])
#
#     distribution = (sns.distplot(database[name].dropna(), ax=ax[1], axlabel=False, vertical=True, rug=False,
#                                  bins=np.arange(0.6, math.ceil(np.max(database[name].dropna())*10)/10, 0.1),
#                                  color='black',
#                                  kde=True, kde_kws={"linestyle": "--", "linewidth": 0.5, "zorder": 3}))
#                                  #fit=stats.norm, fit_kws={"linewidth": 1.0}))
#     # calculate and plot a normal distribution
#     res_mean, res_std = stats.norm.fit(database[name].dropna())
#     lnspc = np.linspace(res_mean-4*res_std, res_mean+4*res_std, 20*len(database[name].dropna()))
#     fit_pdf = stats.norm.pdf(lnspc, res_mean, res_std)
#     ax[1].plot(fit_pdf, lnspc, lw=1, color='black', label='Norm PDF', zorder=4)
#     #calculate and plot a log-normal distribution
#     # log_database = np.log(database[name].dropna())
#     # log_mean = np.mean(log_database)
#     # log_std = np.std(log_database)
#     # shape, loc, scale = stats.lognorm.fit(log_database, floc=0)
#     # x = np.linspace(log_database.min(), log_database.max(), num=400)
#     # plt.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'r', linewidth=3)
#     # n, bins, patches = plt.hist(log_database, bins=5)
#     #
#     # mle, dist_location, dist_scale = stats.lognorm.fit(database[name].dropna(), np.log(res_std), loc=np.log(res_mean))
#     # lgspc = np.linspace(res_mean - 4 * res_std, res_mean + 4 * res_std, 20 * len(database[name].dropna()))
#     # ax[1].plot(fit_pdf, lgspc, lw=1, color='black', label='Norm PDF', zorder=4)
#     ix = []
#     iy = []
#     for j, i in zip(lnspc, fit_pdf):
#         if j >= 1-0.0025:
#                 iy.append(j)
#                 ix.append(i)
#     verts= [(0.0, iy[0]), *zip(ix, iy), (0.0, iy[-1])]
#     poly = Polygon(verts, facecolor='0.8', edgecolor='0.5', alpha=0.6)
#     ax[1].add_patch(poly)
#     Probability = integrate_area(verts)*100
#     chart_text = "%.1f %%" % Probability
#     plt.text(0.1, res_mean, chart_text, rotation=270, verticalalignment='center', fontsize=10, zorder=10)
#     lower_90 = res_mean-1.64*res_std
#     ci_text = "%.3f " % lower_90
#
#     xmin, xmax = ax[1].get_xlim()
#     ax[1].set_xlim([xmin, math.ceil(xmax)+0.01])
#
#     plt.text(xmax, lower_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
#              fontsize=10, rotation='horizontal', zorder=10)
#     higher_90 = res_mean+1.64*res_std
#     ci_text = "%.3f " % higher_90
#     plt.text(xmax, higher_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
#             fontsize=10, rotation='horizontal', zorder=10)
#     ax[1].plot([xmin, xmax+1], [lower_90, lower_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
#     ax[1].plot([xmin, xmax+1], [higher_90, higher_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
#     ax[1].plot([xmin, xmax+1], [1, 1], '-', color='grey', linewidth=2, zorder=2)
#     plt.setp(ax[1].get_yticklabels(), visible=False)
#     plt.setp(ax[1].get_xticklabels(), visible=False)
#     plt.setp(ax[1].set_xticks([]))
#     #plt.xlabel('PDF ', fontsize=10)
#     #ax[1].xlabel(None)
#     quant2 = np.quantile(database[name].dropna(), 0.05, interpolation='nearest')
#     cov = stats.variation(database[name].dropna())
#     info_text = "Mean value: " + str(np.round(res_mean, 3)) + "\nStandard deviation: " + str(np.round(res_std, 3)) \
#                  + "\nCOV: " + str(np.round(cov, 3)) + "\n5% fractile: " + str(np.round(quant2, 3))
#
#     h = ax[0].text(0.0120, 0.97, str(info_text), color='black', horizontalalignment='left',
#                    verticalalignment='top', fontsize=10, rotation='horizontal', zorder=10, transform=ax[0].transAxes,
#                    bbox={'facecolor': 'white', 'edgecolor': 'black'}) # 'pad':10
#     fig_title = name + fabrication + ".png"
#     plt.savefig(fig_title, format='png', dpi=1000) #bbox_inches='tight'
#     fig_title = name + fabrication + ".pdf"
#     plt.savefig(fig_title, format='pdf', dpi=1000) #bbox_inches='tight'
#     fig_title = name + fabrication + ".jpg"
#     plt.savefig(fig_title, format='jpg', dpi=1000)  # bbox_inches='tight'
#     #plt.show()
#     plt.close(fig='all')
#     file_name = name + fabrication + ".txt"
#     stat_file = open(file_name, "w")
#     print("Model compared: \t", name, "\n###################################################",
#           "\nNumber of specimens: \t", len(database[name].dropna()),
#           "\nResiduals Statistics (deviation from proposed model):",
#           "\nMean value: \t", res_mean, "\nStandard error: \t", stats.sem(database[name].dropna()),
#           "\nStandard deviation: \t", res_std, "\n95% Confidence interval \tlower: \t", lower_90,
#           "\tupper: \t", higher_90, "\nProbability of sample being above the limit:", Probability,
#           "\n5% lower quantile: \t", quant2, "\nCOV: \t", cov, file=stat_file)
#     stat_file.close()


def regression_models_red(database, name, fabrication, section_type, save_dir):
    #initiate figure type and shape
    matplotlib.style.use('default')
    inch = 1
    cm = inch/2.54
    fig, ax = plt.subplots(nrows=1, ncols=2,  figsize=[17*cm, 5*cm], sharey='row', constrained_layout=True,
                           gridspec_kw={"wspace": 0.05, 'width_ratios': [5, 1]})
    sns.set_context("paper")
    sns.set_style("whitegrid",
                  rc={'axes.grid': True, 'grid.linestyle': '-', 'grid.color': 'xkcd:light grey', 'xtick.bottom': True,
                      'ytick.left': True, 'marker.size': '0.5', "xtick.direction": "in", "ytick.direction": "in",
                      'font.size': 10, 'font': 'calibri'})
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    ax[0].set_xticks(np.arange(0, 2.6, 0.2))
    ax[0].set_yticks(np.arange(0.6, 2.4, 0.2))
    ax[0].set_xlim([0, 2.401])

    database = database[database['Section Type'] == section_type][database['Fabrication'] == fabrication]
    regression = (sns.regplot(x_ci=90, x="Slenderness", y=name,  # x_bins=np.arange(0, 2.4, 0.2), x_estimator=np.mean,
                              data=database, ax=ax[0], color='black', fit_reg=True, truncate=True, marker="",
                              line_kws={'linewidth': '1.0'}))

    s = database.groupby("Steel Grade").size()
    grades = s.index[s > 0]

    for steel_grade in grades:
        cond_markers = ['s' if steel_grade == 'S960' else '^' if steel_grade == 'S700' else 'o']
        sc = ax[0].scatter(x="Slenderness", y=name, data=database[database['Steel Grade'] == steel_grade],
                      marker=cond_markers[0], label=steel_grade, c='black', zorder=8)
        if cond_markers == ['o']:
            sc.set(facecolors='none', edgecolors='black')

    label_list = ax[0].get_legend_handles_labels()[1]
    label_list.reverse()
    handles_list = ax[0].get_legend_handles_labels()[0]
    handles_list.reverse()
    ax[0].legend(handles_list, label_list, loc=1)

    #sns.lmplot(x="Slenderness", y=name, data=database, hue="Steel Grade")
    # database.groupby('Steel Grade').plot(x='Slenderness', y='JSvsT', kind='scatter', ax=ax[0], subplots=False, style=markers_dict)
    # a = sns.scatterplot(x="Slenderness", y=name, hue="Steel Grade", data=database, hue_order=['S960', 'S700', 'S690'],
    #                 style="Steel Grade", markers=markers_dict, ax=ax[0], legend=False,
    #                 palette=sns.xkcd_palette(["black", "black", "black"]), **{"facecolors": 'none', }, zorder=10)
    # # a=plt.scatter(x="Slenderness", y=name, s=None, c="black", marker='o', linewidths=None, verts=None, facecolor='none',
    # #               edgecolors='black', data=database)

    ax[0].grid(which='major', axis='both', linewidth=0.5, zorder=1)

    ax[0].plot([0, 2.5], [1, 1], linestyle='-', color='grey', linewidth=2, zorder=2)
    plt.rc('font', family='calibri')
    ax[0].set_xlabel(r'Non-dimensional slenderness, ${\overline{\lambda}}$')
    ax[0].set_ylabel(r'$N_{exp}/{N_{Rk}}$')
    distribution = (sns.distplot(database[name].dropna(), ax=ax[1], axlabel=False, vertical=True, rug=False,
                                 bins=np.arange(0.6, math.ceil(np.max(database[name].dropna())*10)/10, 0.1),
                                 color='black',
                                 kde=True, kde_kws={"linestyle": "--", "linewidth": 0.5, "zorder": 3, 'bw': 'scott'}))
                                 # fit=stats.norm, fit_kws={"linewidth": 1.0, "zorder": 4}))

    res_mean, res_std = stats.norm.fit(database[name].dropna())
    lnspc = np.linspace(res_mean-4*res_std, res_mean+4*res_std, 20*len(database[name].dropna()))
    fit_pdf = stats.norm.pdf(lnspc, res_mean, res_std)
    ax[1].plot(fit_pdf, lnspc, lw=1, color='black', label='Norm PDF', zorder=4)
    ix = []
    iy = []
    for j, i in zip(lnspc, fit_pdf):
        if j >= 1-0.0025:
                iy.append(j)
                ix.append(i)
    verts= [(0.0, iy[0]), *zip(ix, iy), (0.0, iy[-1])]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.5', alpha=0.6)
    ax[1].add_patch(poly)
    Probability = integrate_area(verts)*100
    chart_text = "%.1f %%" % Probability
    plt.text(0.1, res_mean, chart_text, rotation=270, verticalalignment='center', fontsize=10, zorder=10)
    lower_90 = res_mean-1.64*res_std
    ci_text = "%.3f " % lower_90
    #set plot limits
    y_lim = math.ceil(np.max(database[name]) * 10) / 10 + 0.1
    y_lim = y_lim if y_lim > 1.4 else 1.4
    xmin, xmax = ax[1].get_xlim()
    ax[1].set_xlim([xmin, math.ceil(xmax)+0.01])
    y_min = math.floor(lower_90 * 10) / 10 - 0.1
    y_min = y_min if y_min < 0.8 else 0.8
    ax[1].set_ylim([y_min, y_lim])
    ax[0].set_ylim([y_min, y_lim])
    if lower_90 >= 0.75:
        plt.text(xmax, lower_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                fontsize=10, rotation='horizontal')
    higher_90 = res_mean+1.64*res_std
    ci_text = "%.3f " % higher_90
    if higher_90 <= np.max(database[name]):
        plt.text(xmax, higher_90, ci_text, color='red', horizontalalignment='right', verticalalignment='bottom',
                fontsize=10, rotation='horizontal', zorder=10)
    ax[1].plot([xmin, xmax+1], [lower_90, lower_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [higher_90, higher_90], '--', color='black', linewidth=1, markersize=5, zorder=10)
    ax[1].plot([xmin, xmax+1], [1, 1], '-', color='grey', linewidth=2, zorder=2)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[1].set_xticks([]))
    #plt.xlabel('PDF ', fontsize=10)
    #ax[1].xlabel(None)

    quant2 = np.quantile(database[name].dropna(), 0.05, interpolation='nearest')
    cov = stats.variation(database[name].dropna())
    info_text = r'$\overline{x}$ = ' + str(np.round(res_mean, 3)) + '\n' + r"$s$ = " + str(np.round(res_std, 3)) \
                + '\n' + r'$COV$ = ' + str(np.round(cov, 3))  # + '\n' + r'$5\%fractile$ = ' + str(np.round(quant2, 3))

    h = ax[0].text(0.0120, 0.97, str(info_text), color='black', horizontalalignment='left',
                   verticalalignment='top', fontsize=10, rotation='horizontal', zorder=10, transform=ax[0].transAxes,
                   bbox={'facecolor': 'white', 'edgecolor': 'black'})  # 'pad':10

    fig_title = save_dir + name + fabrication + section_type + ".png"
    plt.savefig(fig_title, format='png', dpi=1000) #bbox_inches='tight'
    fig_title = save_dir + name + fabrication + section_type + ".pdf"
    plt.savefig(fig_title, format='pdf', dpi=1000) #bbox_inches='tight'
    fig_title = save_dir + name + fabrication + section_type + ".jpg"
    plt.savefig(fig_title, format='jpg', dpi=1000)  # bbox_inches='tight'
    #plt.show()
    plt.close(fig='all')
    file_name = save_dir + name + fabrication + section_type + ".txt"
    stat_file = open(file_name, "w")
    print("Model compared: \t", name, "\nSection type: \t", section_type, "\nFabrication method: \t", fabrication,
          "\nNumber of specimens: \t", len(database[name].dropna()),
          "\n#############################################################################################",
          "\nResiduals Statistics (deviation from proposed model):",
          "\nMean value: \t", res_mean, "\nStandard error: \t", stats.sem(database[name].dropna()),
          "\nStandard deviation: \t", res_std, "\n95% Confidence interval \t lower: \t", lower_90,
          "\t upper: \t", higher_90, "\nProbability of sample being above the limit: \t", Probability,
          "\n5% lower quantile: \t", quant2, "\n", "\nCOV: \t", cov, file=stat_file)
    stat_file.close()


b_curve = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\curve_b\\"
c_curve = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\curve_c\\"
sk = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\SK\\"
main = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\"

i = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\i\\"
b = "C:\\Users\\gabsab\\Documents\\HstrSteel\\Figures\\b\\"

# comparison of different models to evaluate the flexural buckling capacity
# residual stresses according to SK
# regression_models(dbase, "SKKSvsT", sk)

# # residual stresses according to BSK07
regression_models(dbase, "SKvsT", sk)
#
# # EN 1993-1-1 against experimental results
# regression_models(dbase, "ENvsT", c_curve)
#
# # Jönsson & Stan vs experimental results
# regression_models(dbase, "JSvsT", c_curve)
#
# #AISC 360 vs experimental results
# regression_models(dbase, "AISCvsT", main)
#
#
# #comparison of different facbrication methods
# # residual stresses according to SK
# regression_models2(dbase, "SKKSvsT", "Welded", sk)
#
# # residual stresses according to BSK07
# regression_models2(dbase, "SKvsT", "Welded", sk)
#
# # # EN 1993-1-1 against experimental results
# regression_models2(dbase, "ENvsT", "Welded", c_curve)
# regression_models2(dbase, "ENvsT", "Seamless", main)
# regression_models2(dbase, "ENvsT", "Cold-formed", c_curve)
#
# # # Jönsson & Stan vs experimental results
# regression_models2(dbase, "JSvsT", "Welded", c_curve)
# regression_models2(dbase, "JSvsT", "Seamless", main)
# regression_models2(dbase, "JSvsT", "Cold-formed", c_curve)
#
# #AISC 360 vs experimental results
# regression_models2(dbase, "AISCvsT", "Welded", main)
# regression_models2(dbase, "AISCvsT", "Seamless", main)
# regression_models2(dbase, "AISCvsT", "Cold-formed", main)
#
#
# #comparison of different facbrication methods
# #RHS
# regression_models_red(dbase, "JSvsT", "Cold-formed", "RHS", c_curve)
# regression_models_red(dbase, "ENvsT", "Cold-formed", "RHS", c_curve)
# regression_models_red(dbase, "AISCvsT", "Cold-formed", "RHS", main)
#
# regression_models_red(dbase, "JSvsT", "Seamless", "RHS", main)
# regression_models_red(dbase, "ENvsT", "Seamless", "RHS", main)
# regression_models_red(dbase, "AISCvsT", "Seamless", "RHS", main)
#
# regression_models_red(dbase, "ENvsT", "Welded", "RHS", c_curve)
regression_models_red(dbase, "SKvsT", "Welded", "RHS", sk)
# regression_models_red(dbase, "SKKSvsT", "Welded", "RHS", sk)
# regression_models_red(dbase, "AISCvsT", "Welded", "RHS", main)
#
# #regression_models_red(dbase_b, "ENvsT", "Welded", "RHS", b_curve)
# #regression_models_red(dbase_b, "JSvsT", "Welded", "RHS", b_curve)
#
# #I-sections
# regression_models_red(dbase, "JSvsT", "Welded", "I-sections", c_curve)
# regression_models_red(dbase, "ENvsT", "Welded", "I-sections", c_curve)
# regression_models_red(dbase, "SKvsT", "Welded", "I-sections", c_curve)
# regression_models_red(dbase, "AISCvsT", "Welded", "I-sections", main)
#
#
# # regression_models_red(dbase_b, "JSvsT", "Welded", "I-sections", b_curve)
# # regression_models_red(dbase_b, "ENvsT", "Welded", "I-sections", b_curve)
#
#
# dI = dbase.loc[dbase.Fabrication == 'Welded'].loc[dbase['Section Type'] == 'I-sections']
# db = dbase.loc[dbase.Fabrication == 'Welded'].loc[dbase['Section Type'] == 'RHS']
#
#
# #comparison of different facbrication methods
# # residual stresses according to SK
# regression_models2(db, "SKKSvsT", "Welded", b)
#
# # residual stresses according to BSK07
# regression_models2(db, "SKvsT", "Welded", b)
#
# # # EN 1993-1-1 against experimental results
# regression_models2(db, "ENvsT", "Welded", b)
#
# # # Jönsson & Stan vs experimental results
# regression_models2(db, "JSvsT", "Welded", b)
#
# #AISC 360 vs experimental results
# regression_models2(db, "AISCvsT", "Welded", b)
#
# # residual stresses according to BSK07
# regression_models2(dI, "SKvsT", "Welded", i)
#
# # # EN 1993-1-1 against experimental results
# regression_models2(dI, "ENvsT", "Welded", i)
#
# # # Jönsson & Stan vs experimental results
# regression_models2(dI, "JSvsT", "Welded", i)
#
# #AISC 360 vs experimental results
# regression_models2(dI, "AISCvsT", "Welded", i)
