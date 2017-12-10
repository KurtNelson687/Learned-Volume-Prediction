clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');


% Model Switches (choose 1):
            Softmax = 1;  % logistic regression
                LDA = 0;  % Gaussian discriminant analysis
                SVM = 0;  % SVM (doesn't converge for amount of data we have)
Regularized_Softmax = 0;  % regularize with L2 norm ()
 K_Nearest_Neighbor = 0;             
               
               
%% create possible feature sets
% Rename and remove intercept feature
X = X_train(:,2:end);
y = y_train;
y = y';
               
%Adds columns to X so that all second order terms of original features are included
X(:,5) = X(:,1).^2; %square of duration
X(:,6) = X(:,2).^2; %square of front speed
X(:,7) = X(:,3).^2; %square of area

%Adds all two-way interaction terms to X
X(:,8) = X(:,1).*X(:,2); %duration and front speed
X(:,9) = X(:,1).*X(:,3); %duration and area
X(:,10) = X(:,2).*X(:,3); %front speed and area
[mAll, nAll] = size(X);
               
               
%% Select chosen model type
switch modelType
    case 'OLS'
        model = @OLSfit;
    case 'Lasso'
        model = @Lassofit;
    case 'wt_percentDiff'
        model = @wt_percentDiff_fit;
    case 'wt_local'
        model = @wt_local_fit;
    otherwise
        warning('Unexpected model type!')
end

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation
numKfold  = 10;       % if using LOOCV, k=mAll;
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

% Perform feature selection and requires that all of the original features are included
opts = statset('display','iter');

% Perform feature selection employing Hierarchical principle (keep features 1:3 in)
[fs1,history1] = sequentialfs(model,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',nAll,'options',opts);
sse1 = min(history1.Crit);  % sum of squared error from the best fit (sse=mse, since 1 test point)
% Sequential feature selection: performs cross-val on numKfolds, forces
% alg. to keep in the 3 fundamental features, makes sure the alg. performs
% over all possible features (doesn't stop at a local min), and display
% information at each sequential iteration. 


% Perform feature selection with no requirments 
[fs2,history2] = sequentialfs(model,X,y,'cv',numKfold,...
    'nfeatures',nAll,'options',opts);
sse2 = min(history2.Crit); % sum of squared error from the best fit




















