mean = []
std = []

for res_mean, res_std, linetype, n_samples, leg_label, weight in zip(t_distr.t_mean, t_distr.t_st_dev,
                                                                     t_distr.t_specimens, t_distr.LineType,
                                                                     t_distr.Label, t_distr.LineWeight):
    lnspc = np.linspace(res_mean - 4 * res_std, res_mean + 4 * res_std, 200)
    if leg_label in 'MSS':
        leg_label = 'Pr:'
    else:
        leg_label = 'Lk: $\overline{x}$'
    leg_label = leg_label + '\u03BC = ' + str(np.round(res_mean, 3)) + '  \u03C3 = ' + str(np.round(res_std, 3))
    fit_pdf = stats.norm.pdf(lnspc, res_mean, res_std)
    axes.plot(lnspc, fit_pdf, lw=weight, color='black', label=leg_label, linestyle=linetype, zorder=4)
    mean.append(res_mean)
    std.append(res_std)

def Bayes_Gauss_update(mean1=1.0, std1=0.0, n2_samples=1.0, mean2=1.0, std2=0.0):
    #precission1 = 1/std1**2
    #precission2 = 1/std2**2
    #precission_t = precission1+precission2
    #mean_t = (precission1*mean1 + precission2*mean2)/precission_t
    #std_t = np.sqrt(1/precission_t)
    data_range1 = np.linspace(mean1 - 4 * std1, mean1 + 4 * std1, 200)
    data_range2 = np.linspace(mean2 - 4 * std2, mean2 + 4 * std2, 200)
    prior = stats.norm.pdf(data_range1, mean1, std1)
    plt.plot(data_range1, prior, lw=1, color='black', label='MSS', linestyle='-', zorder=4)
    lklh = stats.norm.pdf(data_range2, mean2, std2)
    plt.plot(data_range2, lklh, lw=1, color='black', label='HSS', linestyle='--', zorder=4)
    #post_m = ((std2**2)*mean1 + n2_samples * (std1**2) * mean2) / (n2_samples * (std1**2) + (std2**2))
    #post_var = (std2**2*std1**2) / (n2_samples * std1**2 + std2**2)
    post_var = (1/std1**2+n2_samples/std2**2)**(-1)
    #post_std = np.sqrt(post_var*n2_samples)
    post_std = np.sqrt(post_var)
    post_m = post_std**2*(mean1 / std1**2 + n2_samples * mean2/std2**2)
    data_range_b = np.linspace(post_m - 4 * post_std, post_m + 4 * post_std, 200)
    bayes_pdf = stats.norm.pdf(data_range_b, post_m, post_std)
    leg_label = 'Po:' + '\u03BC = ' + str(np.round(post_m, 3)) + '  \u03C3 = ' + str(np.round(post_std, 3))
    plt.plot(data_range_b, bayes_pdf, lw=1.6, color='black', label=leg_label, linestyle='-', zorder=4)
    #leg_label = 'Posterior - CD2' + '\n\u03BC = ' + str(np.round(post_m, 3)) + '  \u03C3 = ' + str(np.round(post_std, 3))
    #plt.plot(data_range2, bayes_pdf2, lw=1.6, color='white', label=leg_label, linestyle='--', zorder=4)
    # Probability = integrate_area(data_range, bayes_pdf)
    plt.ylim(bottom=0, top=6)
    plt.xlim([0.4, 2.0])
    plt.xlabel(r'${N_u}/{N_{Rk}}$')
    plt.ylabel('PDF')
    plt.legend()
    return post_m, post_std

name = "Bayes_inference_t_all_b"
fig_title = name + ".png"
plt.savefig(fig_title, format='png', dpi=1000) #bbox_inches='tight'
fig_title = name + ".pdf"
plt.savefig(fig_title, format='pdf', dpi=1000) #bbox_inches='tight'
fig_title = name + ".jpg"
plt.savefig(fig_title, format='jpg', dpi=1000) #bbox_inches='tight'

t_distr = lit_data[lit_data["Fabrication"] != "EN_b"]
t_distr = t_distr[t_distr["Fabrication"] != "AISC"]
prior_std = 0.181
prior_mean = 0.944
i=0
n_tot=0
for res_mean, res_std, n_samples in zip(t_distr.t_mean, t_distr.t_st_dev, t_distr.t_specimens):
    if n_samples == 1:
        res_std = prior_std
    n_tot += n_samples
    prior_mean, prior_std = Bayes_Gauss_update(mean1=prior_mean, std1=prior_std,
                                                    n2_samples=n_samples, mean2=res_mean, std2=res_std)
    lnspc_post = np.linspace(prior_mean - 4 * prior_std, prior_mean + 4 * prior_std, 200)
    post = stats.norm.pdf(lnspc_post, loc=prior_mean, scale=prior_std)
    plt.plot(lnspc_post, post, label='Posterior {}'.format(i))
    i += 1



def Bayes_Gauss_update(mean1=1.0, std1=0.0, n2_samples=1.0, mean2=1.0, std2=0.0):
    precission1 = 1/std1**2
    precission2 = 1/std2**2
    precission_t = precission1+precission2
    mean_t = (precission1*mean1 + precission2*mean2)/precission_t
    std_t = np.sqrt(1/precission_t)
    data_range1 = np.linspace(mean1 - 4 * std1, mean1 + 4 * std1, 200)
    data_range2 = np.linspace(mean2 - 4 * std2, mean2 + 4 * std2, 200)
    prior = stats.norm.pdf(data_range1, mean1, std1)
    #plt.plot(data_range1, prior, lw=1, color='black', label='MSS', linestyle='-', zorder=4)
    lklh = stats.norm.pdf(data_range2, mean2, std2)
    #plt.plot(data_range2, lklh, lw=1, color='black', label='HSS', linestyle='--', zorder=4)
    #post_m = ((std2**2)*mean1 + n2_samples * (std1**2) * mean2) / (n2_samples * (std1**2) + (std2**2))
    #post_var = (std2**2*std1**2) / (n2_samples * std1**2 + std2**2)
    #post_var = (1/std1**2+n2_samples/std2**2)**(-1)
    #post_std = np.sqrt(post_var*n2_samples)
    #post_std = np.sqrt(post_var)
    #post_m = post_std**2*(mean1 / std1**2 + n2_samples * mean2/std2**2)
    data_range_b = np.linspace(mean_t - 4 * std_t, mean_t + 4 * std_t, 200)
    bayes_pdf = stats.norm.pdf(data_range_b, mean_t, std_t)
    leg_label = 'Po:' + '\u03BC = ' + str(np.round(mean_t, 3)) + '  \u03C3 = ' + str(np.round(std_t, 3))
    #plt.plot(data_range_b, bayes_pdf, lw=1.6, color='black', label=leg_label, linestyle='-', zorder=4)
    #leg_label = 'Posterior - CD2' + '\n\u03BC = ' + str(np.round(post_m, 3)) + '  \u03C3 = ' + str(np.round(post_std, 3))
    #plt.plot(data_range2, bayes_pdf2, lw=1.6, color='white', label=leg_label, linestyle='--', zorder=4)
    # Probability = integrate_area(data_range, bayes_pdf)
    plt.ylim(bottom=0, top=6)
    plt.xlim([0.4, 2.0])
    plt.xlabel(r'${N_u}/{N_{Rk}}$')
    plt.ylabel('PDF')
    plt.legend()
    return mean_t, std_t


def Bayes_log_update(mean1=1.0, std1=0.0, mean2=1.0, std2=0.0):
    data_range = np.linspace(min(mean1 - 4 * std1, mean2 - 4 * std2), max(mean1 + 4 * std1, mean2 + 4 * std2), 200)
    prior = np.log(stats.norm.pdf(data_range, mean1, std1))
    lklh = np.log(stats.norm.pdf(data_range, mean2, std2))
    post = prior + lklh
    post = np.exp(post)
    plt.plot(data_range, np.exp(prior))
    plt.plot(data_range, np.exp(lklh))
    plt.plot(data_range, post)

def Bayes_log_update_pdf(range1, pdf1, range2, pdf2):
    data_range = np.linspace(min(mean1 - 4 * std1, mean2 - 4 * std2), max(mean1 + 4 * std1, mean2 + 4 * std2), 200)
    prior = np.log(stats.norm.pdf(data_range, mean1, std1))
    lklh = np.log(stats.norm.pdf(data_range, mean2, std2))
    post = prior + lklh
    post = np.exp(post)
    plt.plot(data_range, np.exp(prior), lw=2)
    plt.plot(data_range, np.exp(lklh), lw=4)
    plt.plot(data_range, post, lw3=)


t_distr = lit_data[lit_data["Fabrication"] != "EN_b"]
t_distr = t_distr[t_distr["Fabrication"] != "AISC"]
prior_std = 0.181
prior_mean = 0.944

for res_mean, res_std, n_samples in zip(t_distr.t_mean, t_distr.t_st_dev, t_distr.t_specimens):
    if n_samples == 1:
        res_std = prior_std
    prior_mean, prior_std = Bayes_log_update(mean1=prior_mean, std1=prior_std, mean2=res_mean, std2=res_std)