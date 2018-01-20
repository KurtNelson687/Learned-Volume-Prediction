% Script: BandwidthTesting_wtOLS.m
%
% Author: Kurt Nelson and Sam Maticka
%
% Purpose: This script uses cross-validation to test and identify the best
% bandwidth parameter for local weighted regression.
%%
close all; clear all;
load('../../../DataFiles/data.mat')
addpath('../../Functions/')

%% Variables
bandwidth_all = [1:0.1:6,6:20]; % bandwidths to test
kfold = 10; % number of folds for k-fold cross validation
Xfeatures = [1,2,3,4,7,9,10]; % X features to keep - these were determined from "GeneralRegressionFits.m" using forward selection
%% Setup data
[m,n] = size(X_train);
mround = floor(m/kfold); %rounding to make all folds have the same amount of data

% Extract data so all folds will have the same number of data points, and rename data
X=X_train(1:mround*kfold,2:end);
y=y_train(1:mround*kfold)';

% Adds columns to X so that all second order terms of original features are included
X = addInteractions(X);

X = X(:,Xfeatures); % extract only desired features 

% Create permutation vector
[m,n] = size(X);
perm = randperm(m);

count= 1;
for bandwidth = bandwidth_all
    CV_error = 0; %this will store the sum of squared error
    for i = 1:kfold
        testRange = mround*(i-1)+1:mround*i; %indicies for leave out set
        
        trainRange = 1:m;
        trainRange(testRange) = []; %training indicies
        
        % Organize training and leaveout data
        Xtest_CV = X(perm(testRange),:);
        ytest_CV = y(perm(testRange));
        Xtrain_CV = X(perm(trainRange),:);
        ytrain_CV = y(perm(trainRange));
        
        % Normalize test and training data
        [mTest,nTest]=size(Xtest_CV);
        [mTrain,nTrain]=size(Xtrain_CV);        
        [Xtrain_CV,mu, s] = normalizeVars(Xtrain_CV); % normalize training data 
        for j = 1:nTest %normalize test data
            Xtest_CV(:,j) =  Xtest_CV(:,j)-mu(j);
             Xtest_CV(:,j) =  Xtest_CV(:,j)./s(j);
        end
        
        for k = 1:mTest %loop through test examples and compute the sum or squared error
            % computing weights for example i
            temp = Xtrain_CV-repmat(Xtest_CV(k,:),mTrain,1);
            for j = 1:mTrain
                w(j) = temp(j,:)*temp(j,:)'; % dot product of observations
            end
            w = exp(-w/(2*bandwidth^2)); % compute weights
            
            mdl = fitlm(Xtrain_CV,ytrain_CV,'linear','weights',w);
            ypred = predict(mdl,Xtest_CV(k,:));         % use model to predict on new test data
            CV_error =CV_error +(ytest_CV(k)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error.
        end
    end
    CV_error_all(count) = CV_error/m; % compute MSE CV errror for tested bandwidth
    count = count+1;
end

% bandwidth with minumim CV error
[minVal, minInd] = min(CV_error_all);
minBand = bandwidth_all(minInd);

% Plot CV error as a function of the bandwidth
fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(bandwidth_all,CV_error_all,'linewidth',1)
hold
plot(bandwidth_all(minInd),minVal,'ro','markersize',5)
ylab = ylabel('$CV_{\mathrm{MSE}}$');
set(ylab,'interpreter','Latex','FontSize',10)
xlab = xlabel('$\tau$');
set(xlab,'interpreter','Latex','FontSize',10)
set(gca,'FontSize',8)
print('../Figures/eps/bandWidthTesting','-depsc')
print('../Figures/jpegs/bandWidthTesting','-djpeg','-r600')