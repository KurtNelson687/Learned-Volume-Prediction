clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');


%% Model Switches (choose 1):
Softmax             = 1;  % logistic regression
LDA                 = 0;  % Gaussian discriminant analysis
SVM                 = 0;  % SVM (doesn't converge for amount of data we have)
Regularized_Softmax = 0;  % regularize with L2 norm (). Not available yet.
K_Nearest_Neighbor  = 0;  % KNN using a single neighbor


%% Checks
ModelCheck = [Softmax, LDA, SVM, Regularized_Softmax, K_Nearest_Neighbor];
if sum(ModelCheck)~=1
    error('Choose only 1 model')
end


%% Select chosen model type
if Softmax
    model = @SoftmaxFit;
    Model_Label = 'Softmax';
elseif LDA
    model = @LDAfit;
    Model_Label = 'LDA';
elseif SVM
    model = @SVMfit;
    Model_Label = 'SVM';
elseif Regularized_Softmax
    model = @SoftmaxRegularizedfit;
    Model_Label = 'SoftmaxReg';
elseif K_Nearest_Neighbor
    model = @KNNfit;
    Model_Label = 'KNN';
else
    error('Don''t know what could cause this error')
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
if SVM==0
    numKfold  = 10;       % if using LOOCV, k=m;
    featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)
    
    % Perform feature selection and requires that all of the original features are included
    opts = statset('display','iter');
    
    if Softmax==1
        y = y.*4; % turn into integers
    end
    % Perform feature selection employing Hierarchical principle (keep features 1:3 in)
    [~,history1] = sequentialfs(model,X,y,'cv',numKfold,...
        'keepin',featuresIn,'nfeatures',n,'options',opts);
    sse1 = min(history1.Crit);  % sum of squared error from the best fit (sse=mse, since 1 test point)
    % Sequential feature selection: performs cross-val on numKfolds, forces
    % alg. to keep in the 3 fundamental features, makes sure the alg. performs
    % over all possible features (doesn't stop at a local min), and display
    % information at each sequential iteration.
    
    % Perform feature selection with no requirments
    [~,history2] = sequentialfs(model,X,y,'cv',numKfold,...
        'nfeatures',n,'options',opts);
    sse2 = min(history2.Crit); % sum of squared error from the best fit
    
    % Compute the mean squared error predicted
    [minCV, indMinCV] = min(history1.Crit);
    
    % Extracting features that minimize MSE
    Xfeatures = X(:,history1.In(indMinCV,:));
elseif SVM==1
    Xfeatures = X(:,1:4);
    history1 = [];
    history2 = [];
end


%% Use chosen features to create the chosen model
% split full training data set into training and dev:
[trainInd,~,testInd] = dividerand(length(y_train),.7,0,.3);

Xtrain = Xfeatures(trainInd,:);
ytrain = y(trainInd,:);

Xtest = Xfeatures(testInd,:);
ytest = y(testInd,:);

% Get error of each model and learn a model for use
if Softmax
    CatErr_test  = SoftmaxFit(Xtrain,ytrain,Xtest,ytest);
    CatErr_train = SoftmaxFit(Xtrain,ytrain,Xtrain,ytrain);
    mdl  = mnrfit(Xtrain,ytrain,'model','ordinal');
    prob = mnrval(mdl,Xtest,'model','ordinal'); % n x k
    [~,ypredTest] = max(prob,[],2);
    prob = mnrval(mdl,Xtrain,'model','ordinal'); % n x k
    [~,ypredTrain] = max(prob,[],2);
    
    mdl  = mnrfit(Xfeatures,y,'model','ordinal');
    prob = mnrval(mdl,Xfeatures,'model','ordinal'); % n x k
    [~,ypredTrain_all] = max(prob,[],2);
    Train_allError  = sum(y(:)~=ypredTrain_all(:))/length(y);
elseif LDA
    CatErr_test  = LDAfit(Xtrain,ytrain,Xtest,ytest);
    CatErr_train = LDAfit(Xtrain,ytrain,Xtrain,ytrain);
    mdl = fitcdiscr(Xtrain,ytrain,'DiscrimType','linear'); %  GDA, assumes normal distribution
    ypredTest  = predict(mdl,Xtest);
    ypredTrain = predict(mdl,Xtrain);
    
    mdl = fitcdiscr(Xfeatures,y,'DiscrimType','linear'); %  GDA, assumes normal distribution
    ypredTrain_all = predict(mdl,Xfeatures);
    Train_allError  = sum(y(:)~=ypredTrain_all(:))/length(y);
elseif SVM
    CatErr_test = SVMfit(Xtrain,ytrain,Xtest,ytest);
    CatErr_train = SVMfit(Xtrain,ytrain,Xtrain,ytrain);
    mdl       = fitcecoc(Xtrain,ytrain,'verbose',2);
    ypredTest = predict(mdl,Xtest);           % use model to predict on new test data
    ypredTest = predict(mdl,Xtrain);           % use model to predict on new test data
    
    mdl       = fitcecoc(Xfeatures,y,'verbose',2);
    ypredTrain_all = predict(mdl,Xfeatures); % use model to predict on new test data
    Train_allError  = sum(y(:)~=ypredTrain_all(:))/length(y);
elseif Regularized_Softmax
    
    
elseif K_Nearest_Neighbor
    CatErr_test  = KNNfit(Xtrain,ytrain,Xtest,ytest);
    CatErr_train = KNNfit(Xtrain,ytrain,Xtrain,ytrain); % this should always be 0
    mdl = fitcknn(Xtrain,ytrain,'Distance','euclidean');
    ypredTest  = predict(mdl,Xtest);
    ypredTrain = predict(mdl,Xtrain);
    
    mdl = fitcknn(Xtrain,ytrain,'Distance','euclidean');
    ypredTrain_all = predict(mdl,Xfeatures); % use model to predict on new test data
    Train_allError  = sum(y(:)~=ypredTrain_all(:))/length(y);
end


%% Calculate errors
mtrain = size(Xtrain,1);
mtest  = size(Xtest,1);

trainError = CatErr_train/mtrain;
testError  = CatErr_test/mtest;

%% Save Model
load('Models.mat')
% Initial creation of structure to store model
Models.(sprintf('%s',Model_Label)).Errors.Training   = trainError;
Models.(sprintf('%s',Model_Label)).Errors.Dev        = testError;
Models.(sprintf('%s',Model_Label)).Errors.FullTrain  = Train_allError;
Models.(sprintf('%s',Model_Label)).FullyTrainedModel = mdl;
Models.(sprintf('%s',Model_Label)).HistoryHPrinc     = history1;
Models.(sprintf('%s',Model_Label)).HistoryNoPrinc    = history2;
% Models.(sprintf('%s',Model_Label)).FinalModelPred    = ypredTrain_all;

save('./Models','Models','-append')


%% plot stuff
if Softmax==1
    y = y./4;
    ypredTrain_all = ypredTrain_all./4;
end

if SVM==0
fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(length(featuresIn):n,history1.Crit,'linewidth',1)
hold
plot(1:n,history2.Crit,'linewidth',1)
ylab = ylabel('CV (Logisitc Regression)');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Number of features');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
leg = legend('Enforcing hierarchical principle', 'Unrestricted');
set(leg,'interpreter','Latex','FontSize',6)
print(['./Figures/eps/WriteUp/featureSelection',Model_Label],'-depsc')
print(['./Figures/jpegs/WriteUp/featureSelection',Model_Label],'-djpeg','-r600')
end

videoNum = 1:m;
predLogical = ypredTrain_all(:)~=y(:);

fig2 = figure;%('visible', 'off');
fig2.PaperUnits = 'centimeters';
fig2.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(ypredTrain_all,y,'k.','markersize',15)
hold on;
plot(ypredTrain_all(predLogical),y(predLogical),'r.','markersize',15)

title([Model_Label,'. Training error: ',num2str(Train_allError*100),'%'])
ylab = ylabel('Poured volume (c)');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Predicted volume (c)');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6,'ylim',[0 3])
leg = legend('Correct Classification', 'Incorrect Classification','location','NW');
print(['./Figures/eps/WriteUp/classificationError',Model_Label],'-depsc')
print(['./Figures/jpegs/WriteUp/classificationError',Model_Label],'-djpeg','-r600')

% fig3 = figure;%('visible', 'off');
% fig3.PaperUnits = 'centimeters';
% fig3.PaperPosition = [0 0 8 4];
% set(gca,'box','on')
% 
% plot(y(~predLogical),X(~predLogical,4),'k.','markersize',15)
% hold
% plot(y(predLogical),X(predLogical,4),'r.','markersize',15)
% %
% ylab = ylabel('physics volume');
% set(ylab,'interpreter','Latex','FontSize',8)
% xlab = xlabel('actual volume');
% set(xlab,'interpreter','Latex','FontSize',8)
% set(gca,'FontSize',6)
% leg = legend('correct classification', 'misclassification');
% set(leg,'interpreter','Latex','FontSize',6)
% print('./Figures/eps/WriteUp/classificationError2','-depsc')
% print('./Figures/jpegs/WriteUp/classificationError2','-djpeg','-r600')
%




