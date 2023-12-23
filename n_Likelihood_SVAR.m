% OTPUTS
% n_logLik = negative log likelihood

% INPUTS
% theta = vector of the free elements of A and B (the first elements are the one of A)
% A_res = A matrix restrictions (a matrix with numbers where you want restrictions and nans in the free entries)
% B_res = B matrix restrictions (a matrix with numbers where you want restrictions and nans in the free entries)
% Sigma = covariance matrix
% T = period length
% M = number of variables in the VAR
% params_A = locations of the free elements in the A matrix
% params_B = locations of the free elements in the B matrix

function [n_logLik]=n_Likelihood_SVAR(theta, A_res, B_res, Sigma, T, M, params_A, params_B)
    A = A_res; % setting the A matrix equal to the restricted one
    B = B_res; % setting the B matrix equal to the restricted one

    for c_par = 1 : size(params_A,1)
        A(params_A(c_par,1))=theta(c_par);
    end % putting the element of theta in A

    for c_par = size(params_A,1)+1 : size(params_B,1)+size(params_A,1)
        B(params_B(c_par-size(params_A,1),1))=theta(c_par);
    end % putting the element of theta in B
    
    logLik = -0.5*T*M*(log(2*pi)) - 0.5*T*log((det(pinv(A)*(B*B')*pinv(A)')))-0.5*T*trace((A'*pinv(B)'*pinv(B)*A)*Sigma); % log-likelihood
    n_logLik = -logLik; % negative log-likelihood
end