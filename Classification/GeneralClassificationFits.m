clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');


% Model Switches (choose 1):
Softmax             = 1;  % logistic regression
LDA                 = 0;  % Gaussian discriminant analysis
SVM                 = 0;  % SVM (doesn't converge for amount of data we have)
Regularized_Softmax = 0;  % regularize with L2 norm ()
 K_Nearest_Neighbor = 0;             
               
 % Choose whether to perform error analysis or create a model
 ErrorAnalysis = 1;
 CreateModel   = 0;
 
 
 
%% Checks
Models = [Softmax, LDA, SVM, Regularized_Softmax, K_Nearest_Neighbor];
if sum(Models)~=1
    error('Choose only 1 model')
end

if sum([ErrorAnalysis,CreateModel])~=1
    error('Choose either to analyze or create a model')
end


%% Select chosen model type
if Softmax
    model = @Softmaxfit;
elseif LDA
    model = @LDAfit;
elseif SVM
    model = @SVMfit;
elseif Regularized_Softmax
    model = @SoftmaxRegularizedfit;
elseif K_Nearest_Neighbor
    model = @KNNfit;
else
    warning('Don''t know what causes this warning')
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
% Sequential feature selection: performs cross-val on numKfolds, forces
% alg. to keep in the 3 fundamental features, makes sure the alg. performs
% over all possible features (doesn't stop at a local min), and display
% information at each sequential iteration. 

% % Perform feature selection with no requirments 
% [fs2,history2] = sequentialfs(model,X,y,'cv',numKfold,...
%     'nfeatures',n,'options',opts);
% sse2 = min(history2.Crit); % sum of squared error from the best fit

% Compute the mean squared error predicted
[minCV, indMinCV] = min(history1.Crit);

% Extracting features that minimize MSE
Xfeatures = X(:,history1.In(indMinCV,:)); 


%% Use chosen features to create the chosen model

% split full training data set into training and dev:
[trainInd,~,testInd] = dividerand(length(y_train),.7,0,.3);

Xtrain = Xfeatures(trainInd,:);
ytrain = y(trainInd,:);

Xtest = Xfeatures(testInd,:);
ytest = y(testInd,:);

% Get error of each model, or learn a model for use. 
if Softmax
    if ErrorAnalysis
        CatErr_test  = SoftmaxFit(Xtrain,ytrain,Xtest,ytest);
        CatErr_train = SoftmaxFit(Xtrain,ytrain,Xtrain,ytrain);
    elseif CreateModel
        B = mnrfit(Xtrain,ytrain,'model','ordinal');
        prob = mnrval(B,Xtest,'model','ordinal'); % n x k
        [~,ypredTest] = max(prob,[],2);
        prob = mnrval(B,Xtrain,'model','ordinal'); % n x k
        [~,ypredTrain] = max(prob,[],2);
    end
    
elseif LDA
    if ErrorAnalysis
        CatErr_test  = LDAfit(Xtrain,ytrain,Xtest,ytest);
        CatErr_train = LDAfit(Xtrain,ytrain,Xtrain,ytrain);
    elseif CreateModel
        lda = fitcdiscr(Xtrain,ytrain,'DiscrimType','linear'); %  GDA, assumes normal distribution
        [ypredTest,~,~]  = predict(lda,Xtest);
        [ypredTrain,~,~] = predict(lda,Xtrain);
    end
    
elseif SVM


elseif Regularized_Softmax


elseif K_Nearest_Neighbor


end

mtrain = size(Xtrain,1);
mtest  = size(Xtest,1);

trainError = CatErr_train/mtrain;
testError  = CatErr_test/mtest;

%% Plot stuff












