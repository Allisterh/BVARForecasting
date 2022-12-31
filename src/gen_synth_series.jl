# Generate synthetic data set for testing estimates
# Same series as Ghysels and Marcellino (2018)

n = 500
burn = 100
N = n + burn

using Random, Distributions
Random.seed!(2822882)

# Draw series of white noise shocks
vd = Normal(0,1)
v = rand(vd,N,2)

# Compute correlated shock series
Vmat = [1 0.5; 0 1]
eps = eps * inv(Vmat)

# Compute time series
data = Matrix{Float64}(undef,N,2)
data[1,:] = [0 0]
for i = 2:N
    data[i,1] = 1 + 0.8 * data[i-1,1] + eps[i,1]
    data[i,2] = 1 + 0.5 * data[i-1,2] + 0.6 * data[i-1,1] + eps[i,2]
end