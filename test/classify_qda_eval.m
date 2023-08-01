function [pp, class] = classify_qda_eval(model, F)
% Use:
%   [pp, class] = classify_qda_eval(model, F)
% 
% Input [required]:
%   model.type              Type of discriminant analysis (qda) 
%        .m1                Mean of the first class
%        .m2                Mean of the second class
%        .cov1              Covariance of the first class
%        .cov2              Covariance of the second class
%        .priors            Prior probabilities for the two classes
%     
%   F                       Feature vector [observations X features]
%
%  Output:
%   pp                      Posterior probabilities for the two classes
%   class                   Predicted labels
% 
% See also classify_qda_train, classify_qda_example, classify_train, classify_eval

if strcmpi(model.type, 'qda') == 0
    error('[qda] - The provided model is not a qda');
end

% Parameters extraction from the input model
priors  = model.priors;
m1      = model.m1;
m2      = model.m2;
cov1    = model.cov1;
cov2    = model.cov2;
classes = model.classes;

% Output parameters initialization
NumObservations = size(F, 1);
pp              = zeros(NumObservations, 2);
class           = zeros(NumObservations, 1);

for oId = 1:NumObservations
    
    % Single observation
    x = F(oId, :)';
    
    % Class belonging probabilities
    lh(1) = (1/(sqrt(((2*pi)^length(x))*det(cov1))))*exp(-0.5*(x-m1)'/(cov1)*(x-m1));
    lh(2) = (1/(sqrt(((2*pi)^length(x))*det(cov2))))*exp(-0.5*(x-m2)'/(cov2)*(x-m2));
    %lh(1) = mvnpdf(x, m1, cov1);
    %lh(2) = mvnpdf(x, m2, cov2);
    
    % Posterior class belonging probabilities
    post(1) = lh(1)*priors(1)/(priors(1)*lh(1)+priors(2)*lh(2));
    post(2) = lh(2)*priors(2)/(priors(1)*lh(1)+priors(2)*lh(2));
    
    % Class prediction
    pp(oId,:)  = post;
    [~, idcls] = max(post);
    class(oId) = classes(idcls);   
end