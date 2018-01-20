% Script: BestFit_wtLS.m
%
% Author: Kurt Nelson and Sam Maticka
%
% Purpose: This script test the best fit weighted least squares model on
% the training data that has not been seen by the model by computing the
% test root mean squared error (RMSE). It also computes the test RMSE for
% the physical model for comparision purposes.
%%
%% This script uses 10-fold cross validation to test the bandwidth parameter for local
% weighted regression.
close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');

%% Variable
featuresKeep = [1,2,3,4,5,9,10]; % X features to keep - these were determined from "GeneralRegressionFits.m" using forward selection
bandwidth = 2.2; % best fit bandwidth parameter - determined from "BandwidthTesting_wtOLS.m"

%% Setup data
X_test=X_test(:,2:end);
X_train=X_train(:,2:end);

% Adds interactions terms to training and test data
X_train = addInteractions(X_train); % train data
X_test = addInteractions(X_test); % test data

% Keep only desired data
X_train=X_train(:,featuresKeep);
X_test=X_test(:,featuresKeep);

%% Compute test RMSE for the Physical Prediction
scale = nanmean(y_train'.*X_train(:,4))/nanmean(X_train(:,4).^2); %Scaling for physics prediction
ypred = scale*X_test(:,4);
RMSE_physical  = sqrt(mean((y_test-ypred').^2));

%% Normalize test and training data
[mTest,nTest]=size(X_test);
[mTrain,nTrain]=size(X_train);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

%% Compute test RMSE for weighted least squares model
SSE=0;
for i = 1:mTest %loop through test examples and compute the sum or squared error
    
    % computing weights for example i
    temp = X_train-repmat(X_test(i,:),mTrain,1);
    for j = 1:mTrain
        w(j) = temp(j,:)*temp(j,:)';
    end
    w = exp(-w/(2*bandwidth^2)); % compute weights
    
    mdl = fitlm(X_train,y_train,'linear','weights',w); % fit model
    ypred = predict(mdl,X_test(i,:));         % use model to predict on new test data
    SSE =SSE +(y_test(i)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error. 
end
RMSE_wtLS = sqrt(SSE/mTest);
