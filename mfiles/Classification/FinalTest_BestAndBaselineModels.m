% this script trains our best model (softmax) on all the data tests it on
% the held out test data. The baseline model is also trained and tested on
% the same data.
clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');

%% create possible feature sets
% Rename and remove intercept feature
Xtrain = X_train(:,2:end);
Xtest = X_test(:,2:end);

ytrain = y_train*4;
ytrain = ytrain';
ytest  = y_test*4;
ytest  = ytest';

%Adds columns to X so that all second order terms of original features are included
Xtrain(:,5) = Xtrain(:,1).^2; %square of duration
Xtrain(:,6) = Xtrain(:,2).^2; %square of front speed
Xtrain(:,7) = Xtrain(:,3).^2; %square of area
%Adds all two-way interaction terms to X
Xtrain(:,8) = Xtrain(:,1).*Xtrain(:,2); %duration and front speed
Xtrain(:,9) = Xtrain(:,1).*Xtrain(:,3); %duration and area
Xtrain(:,10) = Xtrain(:,2).*Xtrain(:,3); %front speed and area
% [m, n] = size(Xtrain);

%Adds columns to X so that all second order terms of original features are included
Xtest(:,5) = Xtest(:,1).^2; %square of duration
Xtest(:,6) = Xtest(:,2).^2; %square of front speed
Xtest(:,7) = Xtest(:,3).^2; %square of area
%Adds all two-way interaction terms to X
Xtest(:,8) = Xtest(:,1).*Xtest(:,2); %duration and front speed
Xtest(:,9) = Xtest(:,1).*Xtest(:,3); %duration and area
Xtest(:,10) = Xtest(:,2).*Xtest(:,3); %front speed and area
% [m, n] = size(Xtest);

Xfeatures_train = Xtrain(:,[1:6]); % based on feature selection in GeneralClassificationFits.m
Xfeatures_test  = Xtest(:,[1:6]);


% Baseline Test
scale = sum(ytrain/4)./(sum(Xtrain(:,4))); % Scaling for physics prediction
ypred = scale*Xtest(:,4);

ypred = round(ypred*4);
MeanClassError_phys_test = sum(ypred~=ytest)/length(ytest);


% Train softmax model
mdl  = mnrfit(Xfeatures_train,ytrain,'model','ordinal');
% Test on hold out data
prob = mnrval(mdl,Xfeatures_test,'model','ordinal'); % n x k
[~,ypredTest] = max(prob,[],2);

MeanClassError_test = sum(ytest(:)~=ypredTest(:))/length(ytest);











