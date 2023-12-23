% OUTPUTS
% A = estimated A matrix
% B = estimated B matrix
% SE_A = estimated standard errors of the coefficients of the A matrix (= nan if the element was initially constrained)
% SE_B = estimated standard errors of the coefficients of the B matrix (= nan if the element was initially constrained)
% Sigma = covariance matrix of the innovations (OLS formula)
% IRF = impulse response functions (a matrix of M^2 x HorizonIRF elements)
% PHI_Boot = all the bootstrapped PHIs (4-dimensional object: 1. the variable which responds, 2. the variable's shock, 3. the horizon, 4. the bootstrap repetition)

% INPUTS
% data = table of data (without dates)
% lags = lags of the VAR
% det_comp = deterministic components ('ct' for constant and trend, 'c' for constant, 't' for trend, 'none' for nothing)
% A_res = A matrix restrictions (a matrix with numbers where you want restrictions and nans in the free entries)
% B_res = B matrix restrictions (a matrix with numbers where you want restrictions and nans in the free entries)
% HorizonIRF = temporal length of the IRFs
% repetitions = bootstrap repetitions

function [ A , B , SE_A, SE_B, Sigma , IRF , PHI_Boot ] = SVAR( data , lags , det_comp , A_res, B_res , HorizonIRF , repetitions)

    if ~(strcmp(det_comp, 'ct') || strcmp(det_comp, 't') || strcmp(det_comp, 'c') || strcmp(det_comp, 'none'))
        disp('Unknown deterministic component')
    end % deterministic components selection check
    
    T = size(data,1)-lags; % observations used in the regressions
    M = size(data,2); % number of varaibles
    
    DuplicationMatrix = DuplicationMatrixFunction(M); % calls function that creates duplication matrix
    mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';  % this is the D_plus matrix

    VAR_PI = {}; % empty cell for the VAR coefficients

    for i = 1 : lags
        VAR_PI{i} = nan(M);
    end % creating the matrices of coefficients

    if strcmp(det_comp, 'c')
        VAR_c = nan(M,1);
        VAR_t = zeros(M,1);
    elseif strcmp(det_comp, 'ct')
        VAR_c = nan(M,1);
        VAR_t = nan(M,1);
    elseif strcmp(det_comp, 't')
        VAR_t = nan(M,1);
        VAR_c = zeros(M,1);
    elseif strcmp(det_comp, 'none')
        VAR_c = zeros(M,1);
        VAR_t = zeros(M,1);
    end % preparing the matrices for the deterministic components
    VAR = varm('Constant',VAR_c,'AR',VAR_PI,'Trend',VAR_t); % creating the var model


    [EstVAR,EstSE,logLikVAR,Residuals] = estimate(VAR,data,'Display',"full"); % Estimation of the VAR by ML

    Sigma = EstVAR.Covariance; % estimated covariances matrix of the innovations
    
    StructuralParam = sum(sum(isnan(A_res)))+sum(sum(isnan(B_res))); % dimesion of vector of structural parameters (beta in the slides)
    
    InitialValues=randn(StructuralParam,1)/10; % initial values in order to enter the likelihood maximization
    
    params_A=find(isnan(A_res)); % locations of the parameters to estimate in A (the nans in A_res)
    params_B=find(isnan(B_res)); % locations of the parameters to estimate in B (the nans in B_res)
    
    options = optimset('MaxFunEvals',200000,'TolFun',1e-500,'MaxIter',200000,'TolX',1e-100); % optimization settings
    fun = @(theta) n_Likelihood_SVAR(theta, A_res, B_res, Sigma, T, M, params_A, params_B); % function to minimize (n_Likelihood_SVAR), @(theta) indicates that what the function can control is theta, while keeps the other inputs as given
    [StructuralParam_Estimation_MATRIX, n_logLik,exitflag,output,grad, Hessian_MATRIX] = fminunc(fun, InitialValues, options); % StructuralParam_Estimation_MATRIX is the vector of the parameters that maximize the log-likelihood, n_logLik is the minimized negative log-likelihood, Hessian_MATRIX is the Hessian matrix
    
    SE_Hessian_MATRIX = diag(inv(Hessian_MATRIX)).^0.5; % the S.E. of the coefficients
    
    A = A_res; % settings A equal to the restricetd A
    SE_A = nan(size(Sigma,1)); % empty matrix for the standard errors of A
    HSelection_A = zeros(M*M,size(params_A,1)); % empty matrix for S_A
    
    B = B_res; % settings B equal to the restricetd B 
    SE_B = nan(size(Sigma,1)); % empty matrix for the standard errors of B
    HSelection_B = zeros(M*M,size(params_B,1)); % empty matrix for S_B
    
    for c_par = 1 : size(params_A,1) % loop that fills the matrices for A
        A(params_A(c_par,1)) = StructuralParam_Estimation_MATRIX(c_par);
        SE_A(params_A(c_par,1)) = SE_Hessian_MATRIX(c_par);
        HSelection_A(params_A(c_par,1),c_par) = 1;
    end
    
    for c_par = size(params_A,1)+1 : size(params_B,1)+size(params_A,1) % loop that fills the matrices for B
        B(params_B(c_par-size(params_A,1),1))=StructuralParam_Estimation_MATRIX(c_par);
        SE_B(params_B(c_par-size(params_A,1),1)) = SE_Hessian_MATRIX(c_par);
        HSelection_B(params_B(c_par-size(params_A,1),1),c_par-size(params_A,1)) = 1;
    end
    
    for i = 1:M % signs normalization
        if B(i,i)<0
            B(:,i)=-B(:,i);
        end
        if A(i,i)<0
            A(:,i)=-A(:,i);
        end
    end

    Likelihood_SVAR = -1*n_logLik; % log-likelihood of SVAR
    LR_test_overid = -2*(Likelihood_SVAR - logLikVAR); % LR statistics
    df = M*(M+1)/2 -size(params_A,1)-size(params_B,1); % degrees of freedom
    if df>0
        PVal = 1-chi2cdf(LR_test_overid,df);
        disp(['The p-value of the overidentification restrictions is ' num2str(PVal)])
    end % display the p-value of the overidentification restrictions if there is overidentification (i.e. the number of degrees of freedom is positive)

    if rank(2*mDD*[-kron(pinv(A)*B*B'*pinv(A)',pinv(A)),kron(pinv(A)*B,pinv(A))]*[HSelection_A,zeros(M^2,size(params_B,1));zeros(M^2,size(params_A,1)),HSelection_B])==size(params_A,1)+size(params_B,1) % rank Jacobian == number of parameters
        disp('The necessary and sufficient rank condition for local identification is satisfied')
    else
        disp('The necessary and sufficient rank condition for local identification is not satisfied')
    end % necessary and sufficient rank condition check
    
    J=[eye(M) zeros(M,M*(lags-1))]; % selection matrix
    CompanionMatrix = []; %empty companion matrix
    
    for p = 1:lags
        CompanionMatrix = [CompanionMatrix EstVAR.AR{p}];
    end % loop that fills the companion matrix with the estimated PIs
    
    CompanionMatrix=[CompanionMatrix;eye(M*(lags-1)) zeros(M*(lags-1),M)]; % adding the other elements of the companion matrix
    
    for h = 0 : HorizonIRF
        PHI(:,:,h+1)=J*CompanionMatrix^h*J'*pinv(A)*B;
    end % computing all the phis
    
    for h = 0 : HorizonIRF
        for i=1:M
            for j=1:M
                IRF(h+1,M*(i-1)+j)=PHI(i,j,h+1);
            end
        end
    end % creating the IRFs matrix

    %% BOOTSTRAP

    Residuals_B=[]; % empty residuals matrix
    data_B=zeros(T+lags,M*repetitions); % empty bootstrap data matrix
    data_B(1:lags,:)=kron(ones(1,repetitions),data(1:lags,:)); % setting the first elements equal to the original data
    
    for boot = 1 : repetitions
        TBoot=datasample(1:T,T); % resample from 1 to T
        Residuals_B=[Residuals_B Residuals(TBoot,:)];  % bootstrap errors 
    end % this loop generates the M residuals series 'repetitions' times, and put all of them in a T x M*repetitions matrix
    
    for t=1+lags:T+lags
        data_h=zeros(1,size(data_B,2)); % empty row of size 1 x M*repetitions
        for p=1:lags
            data_h=data_h+data_B(t-p,:)*kron(eye(repetitions),EstVAR.AR{p}');
        end % loop that computes the predicted obs of t+1 (without the error and the deterministic components)
        data_B(t,:)=data_h+kron(ones(1,repetitions),EstVAR.Constant')+kron(ones(1,repetitions),EstVAR.Trend')*t+Residuals_B(t-lags,:); % adding the error and the deterministic components
    end % loop that creates the bootstrap M series 'repetitions' times and put them in a T x M*repetitions
    
    data_B_all={}; % empty cell
    for boot = 1 : repetitions
        data_B_all{boot}=data_B(:,1+(boot-1)*M:M+(boot-1)*M);
    end % loop that puts all the different bootstrap M series in the empty cell created above (in each cell we will have the M different bootstrap series, and we will have 'repetitions' cell)
    
    for boot = 1 : repetitions 
        disp(boot)
        data_B=data_B_all{boot}; % choosing one of the bootstrapped dataset
        
        % from now to the end of the loop, the code is identical to the one
        % we used above for estimation (we eliminate some lines because
        % they are not needed for the bootstrap), and the process is
        % repeated 'repetitions' times, using each time a different
        % bootstrap dataset

        [EstVAR_B] = estimate(VAR,data_B); % we use the same VAR defined above, in order to estimate the same model
    
        Sigma_B=EstVAR_B.Covariance;
    
        options = optimset('MaxFunEvals',200000,'TolFun',1e-500,'MaxIter',200000,'TolX',1e-100,'Display', 'off');   
        fun = @(theta) n_Likelihood_SVAR(theta, A_res, B_res, Sigma_B, T, M, params_A, params_B);
        [StructuralParam_Estimation_MATRIX_B] = fminunc(fun, InitialValues, options);
    
        A_B = A_res;
        B_B = B_res;
    
        for c_par = 1 : size(params_A,1)
            A_B(params_A(c_par,1)) = StructuralParam_Estimation_MATRIX_B(c_par);
        end
    
        for c_par = size(params_A,1)+1 : size(params_B,1)+size(params_A,1)
            B_B(params_B(c_par-size(params_A,1),1))=StructuralParam_Estimation_MATRIX_B(c_par);
        end
    
        for i = 1:M
            if B_B(i,i)<0
                B_B(:,i)=-B_B(:,i);
            end
            if A_B(i,i)<0
                A_B(:,i)=-A_B(:,i);
            end
        end
    
        J=[eye(M) zeros(M,M*(lags-1))];
        CompanionMatrix_Boot = [];
    
        for p = 1:lags
            CompanionMatrix_Boot = [CompanionMatrix_Boot EstVAR_B.AR{p}];
        end
    
        CompanionMatrix_Boot=[CompanionMatrix_Boot;eye(M*(lags-1)) zeros(M*(lags-1),M)];
    
        for h = 0 : HorizonIRF
            PHI_Boot(:,:,h+1,boot)=J*CompanionMatrix_Boot^h*J'*pinv(A_B)*B_B;
        end
    end 
end 