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
featuresKeep = [1,2,3,4,5,9,10];

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

%%%%%%%%%%%%%Physical Prediction
scale = nanmean(y_train'.*X_train(:,4))/nanmean(X_train(:,4).^2); %Scaling for physics prediction
ypred = scale*X_test(:,4);
RMSE_physical  = sqrt(mean((y_test-ypred').^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[mTest,nTest]=size(X_test);
[mTrain,nTrain]=size(X_train);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

SSE=0;
for i = 1:mTest %loop through test examples and compute the sum or squared error
    
    % computing weights for example i
    temp = X_train-repmat(X_test(i,:),mTrain,1);
    for j = 1:mTrain
        w(j) = temp(j,:)*temp(j,:)';
    end
    w = exp(-w/(2*bandwidth^2));
    
    mdl = fitlm(X_train,y_train,'linear','weights',w);
    ypred = predict(mdl,X_test(i,:));         % use model to predict on new test data
    SSE =SSE +(y_test(i)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error. 
end
RMSE_wtLS = sqrt(SSE/mTest);
