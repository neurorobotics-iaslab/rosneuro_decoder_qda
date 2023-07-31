function model = classify_qda_train(F, Fk, varargin)
% Use:
%   model = classify_qda_train(F, Fk [, varargin])
% 
% Input:
%   F                       Feature vector, [observations X features]
%   Fk                      Class labels, [observations X 1]. Only two
%                           classes allowed
% Optional:
%   'priors', [val1 val2]   prior probabilities for the two classes
%   'estimation', 'type'    Covariance estimation:
%                           - 'estimation', 'none'                  
%                               No estimation [DEFAULT]
%                           - 'estimation', 'shrink', 'lambda', val
%                               Shrinkage of cavariance matrices with lambda
%                               parameter
% 
% Output:
%   model.type              Type of discriminant analysis (qda) 
%        .m1                Mean of the first class
%        .m2                Mean of the second class
%        .cov1              Covariance of the first class
%        .cov2              Covariance of the second class
%        .priors            Prior probabilities for the two classes
%        .estimation        Estimation type ('none', 'shrink')
%        .lambda            Lambda parameter for shrinkage
% 
% See also classify_qda_eval, classify_qda_example, classify_train,
% classify_eval


    %% Handling input arguments
    if nargin < 2
        error('[qda] - Error. Not enogh input arguments');
    end
    
    def_priors      = [0.5 0.5];
    def_estimation  = 'none';
    def_lambda      = nan;
    
    pnames  = {'priors',  'estimation',   'lambda'};
    default = {def_priors, def_estimation, def_lambda};
    
    %[~, msg, priors, estimation, lambda] = util_getargs(pnames, default, varargin{:});
    msg = '';
    priors =  [0.5 0.5];
    estimation = 'shrink';
    lambda = 0.5;
    
    if isempty(msg) == false
        error(['[qda] - ' msg]);
    end
    
    if isequal(sum(priors), 1) == false
        error('[qda] - The provided priors do not sum to 1');
    end

    %% Dataset creations
    classes = unique(Fk);
    if isequal(length(classes), 2) == false
        error('[qda] - Number of classes must be equal to 2');
    end
    
    % Classes mean computation
    m1 = mean(F(Fk == classes(1),:))';
    m2 = mean(F(Fk == classes(2),:))';
    
    % Classes covariance computation
    cov1 = cov(F(Fk == classes(1),:));
    cov2 = cov(F(Fk == classes(2),:));

    % Estimation of the covariance
    switch estimation
        case 'none'         
        case 'shrink'
            if size(F, 2) == 1
                warning('[qda] - Skipping regularization. Cannot shrink covariance with only one feature');
                estimation = 'none';
                lambda     = nan;
            else
                cov1  = (1 - lambda)*cov1 + (lambda/size(F, 2))*trace(cov1)*eye(size(cov1));
                cov2  = (1 - lambda)*cov2 + (lambda/size(F, 2))*trace(cov2)*eye(size(cov2));
            end
        otherwise
            error('[qda] - Unknown covariance type');
    end
    
    model.type       = 'qda';
    model.m1         = m1;
    model.m2         = m2;
    model.cov1       = cov1;
    model.cov2       = cov2;
    model.classes    = classes;
    model.priors     = priors;
    model.estimation = estimation;
    model.lambda     = lambda;

end
