'''
EVT Modeling Demonstration
'''

# IMPORTS
import scipy.stats as stats
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
fig, ax = plt.subplots(1, 1)

# MOCK DATA GENERATION
sample_size = 10000
# Generates and samples from two Gaussian distributions
l = [i for i in stats.norm.rvs(size=sample_size, loc=10, scale=10) if i >= 0]
#plt.hist(l, bins=30, alpha=0.8, density=True)
r = stats.norm.rvs(size=sample_size, loc=50, scale=15)
#plt.hist(r, bins=100, alpha=0.8)
#plt.show()

# EVT MODELING FUNCTION
u = 90
nu = len([i for i in r if i > u])
n = len(r)

x_pdf, edges = np.histogram(r, bins=100, density=True)
x_pdf /= sum(x_pdf)
bin_center = (edges[:-1]+edges[1:])/2
x_cdf = np.cumsum(x_pdf)
plt.plot(bin_center, x_pdf)
plt.plot(bin_center, x_cdf)
#print(bin_center)
#print(x_cdf)

'''
fu_pdf = np.histogram([i-u for i in l if i > u], bins=30)[0]/nu

plt.plot(fu_empirical)

fu_empirical_cdf = [0] * len(fu_empirical)
fu_empirical_cdf[0] = fu_empirical[0]
for i in range(1, len(fu_empirical_cdf)):
    fu_empirical_cdf[i] = fu_empirical_cdf[i-1] + fu_empirical[i]

plt.plot(fu_empirical_cdf)
'''

def f_empirical(x, x_cdf):
    for possible_bin in range(1, len(bin_center)):
        if bin_center[possible_bin] >= x:
            binsize = (bin_center[possible_bin] - bin_center[possible_bin-1])
            ratio = (x - bin_center[possible_bin-1])/binsize
            #print(f'binsize={binsize}, ratio={ratio}')
            #return (1-ratio)*x_cdf[possible_bin-1] + (ratio)*x_cdf[possible_bin]
            return x_cdf[possible_bin-1]
    return 1

fu = []
for x in range(u, 120):
    y = x-u
    #print(f_empirical(y+u, x_cdf), f_empirical(u, x_cdf))
    fu.append(
        (f_empirical(y+u, x_cdf) - f_empirical(u, x_cdf)) / (1 - f_empirical(u, x_cdf))
    )
plt.plot(range(u, 120), np.array(fu))

# CHANGE THIS LATER
fake_data = []
fu_pdf = [0] * len(fu)
fu_pdf[0] = fu[0]
for i in range(1, len(fu_pdf)):
    fu_pdf[i] = fu[i] - fu[i-1]
for i in range(len(fu_pdf)):
    fake_data += [i for _ in range(int(fu_pdf[i] * 1000))]

'''
gpd = lambda x, c: stats.genpareto(c).cdf(x)
c = opt.curve_fit(gpd, range(u, 120), fu)[0][0]
'''

c = stats.genpareto.fit(fake_data, floc=0, fscale=1)[0]
print(f"Optimized GPD Parameter: {c}")
plt.plot(range(u, 120), stats.genpareto.cdf(range(len(fu)), c))

plt.legend(["PDF", "CDF_Emp", "Fu_CDF_Emp", "Fu_GPD"])
plt.show()
#stats.genpareto.fit()