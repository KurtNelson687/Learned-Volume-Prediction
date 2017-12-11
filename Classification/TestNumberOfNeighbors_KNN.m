% this script compares the error for using a different number of neighbors
% to compare against. 1 neighbor is the lowest error
clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

for i=1:4
    if i==1
        model = @KNNfit;
    elseif i==2
        model = @KNNfit2;
    elseif i==3
        model = @KNNfit4;
    elseif i==4
        model = @KNNfit6;
    end


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
[m, n] = size(X);

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation

numKfold  = 10;       % if using LOOCV, k=m;
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)

% Perform feature selection and requires that all of the original features are included
opts = statset('display','iter');

% Perform feature selection employing Hierarchical principle (keep features 1:3 in)
[fs1,history1] = sequentialfs(model,X,y,'cv',numKfold,...
    'keepin',featuresIn,'nfeatures',n,'options',opts);
sse1 = min(history1.Crit);  % sum of squared error from the best fit (sse=mse, since 1 test point)


% Compute the mean squared error predicted. the lowest of the 4 is the k chosen
minCV(i) = min(history1.Crit);


end












