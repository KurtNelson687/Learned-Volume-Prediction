%% This script uses 10-fold cross validation to test the bandwidth parameter for local
% weighted regression.

close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');
numTrees = 150;
maxNumSplits = 100;
learningRate = 0.1;

Xfeatures = 1:10;%[1,2,3,4];
X=X_train(:,2:end);
y=y_train';

%Adds columns to X so that all second order terms of original features are included
X(:,5) = X(:,1).^2; %square of duration
X(:,6) = X(:,2).^2; %square of front speed
X(:,7) = X(:,3).^2; %square of area

%Adds all two-way interaction terms to X
X(:,8) = X(:,1).*X(:,2); %duration and front speed
X(:,9) = X(:,1).*X(:,3); %duration and area
X(:,10) = X(:,2).*X(:,3); %front speed and area

X = X(:,Xfeatures);
[m,n] = size(X);
t = templateTree('MaxNumSplits',maxNumSplits);
Mdl = fitensemble(X,y,'LSBoost',numTrees,t,...
    'Type','regression','KFold',10,'LearnRate',learningRate);
MSE = kfoldLoss(Mdl);
%ypred = predict(Mdl,X(1,:)); 