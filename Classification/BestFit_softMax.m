%% This script uses 10-fold cross validation to test the bandwidth parameter for local
% weighted regression.
close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');
bandwidth = 2.2;

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation
numKfold = 10;      % LOOCV: k=m
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)
opts = statset('display','iter'); % displays iterations
featuresKeep = [1,2,3,4,5,6];

X_test=X_test(:,2:end);
X_train=X_train(:,2:end);
%Adds columns to X so that all second order terms of original features are included
X_train(:,5) = X_train(:,1).^2; %square of duration
X_train(:,6) = X_train(:,2).^2; %square of front speed
X_train(:,7) = X_train(:,3).^2; %square of area
X_train(:,8) = X_train(:,1).*X_train(:,2); %duration and front speed
X_train(:,9) = X_train(:,1).*X_train(:,3); %duration and area
X_train(:,10) = X_train(:,2).*X_train(:,3); %front speed and area

X_test(:,5) = X_test(:,1).^2; %square of duration
X_test(:,6) = X_test(:,2).^2; %square of front speed
X_test(:,7) = X_test(:,3).^2; %square of area
X_test(:,8) = X_test(:,1).*X_test(:,2); %duration and front speed
X_test(:,9) = X_test(:,1).*X_test(:,3); %duration and area
X_test(:,10) = X_test(:,2).*X_test(:,3); %front speed and area

X_train=X_train(:,featuresKeep);
X_test=X_test(:,featuresKeep);
y_test = y_test.*4;
y_train = y_train.*4;
% fit coefficients, B for a multinomial logistic regression model
B = mnrfit(X_train,y_train,'model','ordinal'); 
% assumes natural ordering among the response (ytrain) categories.
% B is (p+1)x(k?1). includes intercept.

% calculates probability for each observation to be 1 of k categories.
prob = mnrval(B,X_test,'model','ordinal'); 
% prob is nxk

% Choose category with highest probability
[~,ypred] = max(prob,[],2); % column index corresponds to category index

% error: ratio of wrongly categorized data
missclassError = sum(y_test(:)~=ypred(:))/length(y_test)
