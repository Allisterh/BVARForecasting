module BVARForecasting

using LinearAlgebra

function var_ols(data,p)
    """
    Inputs: data - matrix of data, p - number of lags
    Outputs: Γ_hat - OLS regression coefficients, S - 
    See Karlsson (2013), Section 2.1.1
    """
    # Dimensions
    Tp, m = size(data)
    T = Tp - p # Time obs in sample
    d = 1  # deterministic variables (constants)
    k = m*p+d

    # Matrix with lag structure
    Y = data[(p+1):Tp,:]
    Z = zeros(T,k);
    for lag in 1:p 
        Z[:,(m*(lag-1)+1):(m*lag)] = data[(p+1-lag):(Tp-lag),:]
    end
    Z[:,m*p+1] = ones(T);
    #return Y, Z

    # Least squares formula
    #\Gamma_hat k x m matrix of coefficients
    Γ_hat = inv(Z'Z)Z'Y
    uhat = (Y - Z*Γ_hat)
    S = uhat' * uhat #/ (T - m*p - 1)
    return Γ_hat, S, Y, Z
end

mutable struct BVARModel
    gamma_sample
    psi_sample
    u_sample
    y_til_sample
    data
end

function simulate_norm_wish_post(data, lags, sam_size)
    """
    # Algorithm 1
    # Simulating the predictive distribution with a normal-Wishart posterior
    # Inputs - sample size, forecast length, data, deterministic vars
    # outputs - vector of data with simulated future values    
    """        
    # Check that data is in appropriate format
    
    Γ_hat, S, Y, Z = var_ols(data, lags)

    invZZ = Hermitian(inv(Z'Z))
    S_chol = cholesky(Hermitian(S))

    Ψ_sam = rand(InverseWishart(T-k,S_chol),sam_size)
    Γ_sam = Vector(undef, sam_size)
    u_sam = Array{Float64}(undef, sam_size, forc_len, m)
    y_til = Vector(undef, sam_size)

    for i in 1:sam_size
        # Draw samples
        Γ_sam[i] = rand(MatrixNormal(Γ_hat, invZZ, Ψ_sam[i]),1)[1]
        for h in 1:forc_len
            u_sam[i,h,:] = rand(MvNormal(zeros(m), Hermitian(Ψ_sam[i])),1)
        end

        # Combine into forecast
        Y_forc = [Y; zeros(forc_len,m)]
        Z_forc = [Z; zeros(forc_len,k)]
        for h in 1:forc_len
            Z_forc[T+h, 1:m] = Y_forc[T+h-1,1:m]
            Z_forc[T+h, (m+1):(m*p)] = Z_forc[T+h-1,1:(m*(p-1))]
            Z_forc[T+h, (m*p+1):(m*p+d)] = ones(d)
            Y_forc[T+h,:] = Z_forc[T+h-1,:]' * Γ_sam[i] + u_sam[i,h,:]'
        end
        y_til[i] = Y_forc[(T+1):(T+forc_len),:]
    end

    # Reurn sample of draws from the forecast distribution
    return BVARModel(Γ_sam, Ψ_sam, u_sam, y_til)
end

end  # end module
