clear;clc;close all;
load('FitData_usable.mat')

% create possible feature sets
% Rename and remove intercept feature
X=Xgood(:,2:end);
y=ygood;
y = y';
y = y*4; % make an integer from 1 to k

%Adds columns to X so that all second order terms of original features are included
X(:,5) = X(:,1).^2; %square of duration
X(:,6) = X(:,2).^2; %square of front speed
X(:,7) = X(:,3).^2; %square of area

%Adds all two-way interaction terms to X
X(:,8) = X(:,1).*X(:,2); %duration and front speed
X(:,9) = X(:,1).*X(:,3); %duration and area
X(:,10) = X(:,2).*X(:,3); %front speed and area
[mAll, nAll] = size(X);

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation
tic
numKfold = mAll;      % LOOCV: k=m
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

%Perform feature selection and require that all of the original features are included
opts = statset('display','iter');
[fs1,history1] = sequentialfs(@SVMfit,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit); % sum of squared error from the best fit (sse=mse, since 1 test point)
% Sequential feature selection: performs cross-val on numKfolds, forces
% alg. to keep in the 3 fundamental features, makes sure the alg. performs
% over all possible features (doesn't stop at a local min), and display
% information at each sequential iteration. 

%Perform feature selection with no requirments
[fs2,history2] = sequentialfs(@SVMfit,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit
toc
% sse1 = . sse2 = .
%% choose features based on selection above
% Xfeatures = X(:,[1;2;3;6;10]); %Extracting only features we want
% second lowest sse yielding feature set chosen to not overfit


