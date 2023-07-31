%% Example code to use QDA classifier
clc;clearvars;close all;

npat = 1000;

%% create synthetic data
disp(['Creating new data, so after you need to run LDA classifier with rosneuro and to compare the results run only the final' ...
    'three sections']);
r1 = 0.5 + 0.95*randn(npat,2);
r2 = 3.5 + 0.95*randn(npat*2,2);
data = [r1;r2];
labels = [ones(1,size(r1,1)) 2*ones(1,size(r2,1))];

%% train QDA
model = classify_qda_train(data,labels,'qda');

%% test QDA
[pp, res] = classify_qda_eval(model,data);

accuracy  = 100*sum(res'==labels)/length(labels);

%% graphics
figure(1)
plot(data(labels==1,1),data(labels==1,2),'.') ; hold on;
plot(data(labels==2,1),data(labels==2,2),'gx');
axis([0 6 0 6]);
mis_ind = find(res'~=labels);
plot(data(mis_ind,1),data(mis_ind,2),'ro','markersize',10);
legend('class-1', 'class-2', 'misclassified');
hold off


%% Save data
writematrix(data, '/home/paolo/rosneuro_ws/src/rosneuro_decoder_qda/test/features.csv');

%% Load rosneuro probabilities
load('/home/paolo/rosneuro_ws/src/rosneuro_decoder_qda/test/output.csv');

%% See differences
diff = max(abs(pp - output), [], 'all');
disp(['max difference: ' num2str(diff)]);