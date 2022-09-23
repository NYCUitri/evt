'''
EVT Modeling Demonstration
'''

# IMPORTS
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Naive CDF Estimation
def f_empirical(x, x_cdf):
    for possible_bin in range(1, len(bin_center)):
        if bin_center[possible_bin] >= x:
            binsize = (bin_center[possible_bin] - bin_center[possible_bin-1])
            ratio = (x - bin_center[possible_bin-1])/binsize
            #print(f'binsize={binsize}, ratio={ratio}')
            return (1-ratio)*x_cdf[possible_bin-1] + (ratio)*x_cdf[possible_bin]
    return 1

# Iteration through different sample sizes
for sample_size in [1000, 2000, 5000, 50000]:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f"Sample Size: {sample_size}")
    print(f"Sample Size: {sample_size}")

    # MOCK DATA GENERATION
    # Generates and samples from two Gaussian distributions
    #l = [i for i in stats.norm.rvs(size=sample_size, loc=10, scale=10) if i >= 0]
    #plt.hist(l, bins=30, alpha=0.8, density=True)
    r = stats.norm.rvs(size=sample_size, loc=50, scale=15)
    #plt.hist(r, bins=100, alpha=0.8)

    # EVT MODELING FUNCTION
    u = 80
    print(f"Threshold: {u}")
    nu = len([i for i in r if i > u])
    n = len(r)
    print(f"Number of exceedances: {nu}")

    x_pdf, edges = np.histogram(r, bins=100, density=True)
    x_pdf /= sum(x_pdf)
    bin_center = (edges[:-1]+edges[1:])/2
    x_cdf = np.cumsum(x_pdf)
    ax[0].plot(bin_center, x_cdf) # F_empirical

    fu = []
    for x in range(u, 120):
        y = x-u
        fu.append(
            (f_empirical(y+u, x_cdf) - f_empirical(u, x_cdf)) / (1 - f_empirical(u, x_cdf))
        )
    ax[1].plot(range(u, 120), np.array(fu)) # F_u_empirical

    fake_data = []
    fu_pdf = [0] * len(fu)
    fu_pdf[0] = fu[0]
    for i in range(1, len(fu_pdf)):
        fu_pdf[i] = fu[i] - fu[i-1]
    for i in range(len(fu_pdf)):
        fake_data += [i for _ in range(int(fu_pdf[i] * 1000))]

    c, loc, scale = stats.genpareto.fit(fake_data)
    print(f"Optimized GPD Parameters: {c}, {loc}, {scale}")
    fu_cdf_gpd = stats.genpareto.cdf(range(len(fu)), c, loc, scale)
    ax[1].plot(range(u, 120), fu_cdf_gpd) # F_u_GPD

    f_cdf = (1 - (n - nu)/n) * fu_cdf_gpd + (n - nu)/n
    #print("CDF Values:")
    #print(f_cdf)
    ax[0].plot(range(u, 120), f_cdf) # F_GPD

    # Calculate MSE
    mse = 0
    for i in range(len(f_cdf)):
        mse += (f_cdf[i] - f_empirical(i+u, x_cdf))**2
    mse /= len(f_cdf)
    print(f"MSE from Empirical: {mse}")

    '''
    f_pdf = []
    for x in range(len(f_cdf)-1):
        f_pdf.append(f_cdf[x+1] - f_cdf[x])
    ax[1].plot(range(u+1, 120), f_pdf)
    '''

    ax[0].legend(["F_Emp", "F_GPD"])
    ax[1].legend(["Fu_Emp", "Fu_GPD"])

    print(f"Writing to: gpd_{sample_size}.png")
    fig.savefig(f"gpd_{sample_size}.png")
    print("-------------------------------")
    #plt.show()