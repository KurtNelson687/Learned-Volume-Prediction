%% This script uses 10-fold cross validation to test the bandwidth parameter for local
% weighted regression.

close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');

kfold = 10;
Xfeatures = [1,2,3,4,7,9,10];
[m,n] = size(X_train);
mround = floor(m/kfold); %rounding to make all k=10 folds have the same amount of data
X=X_train(1:mround*kfold,2:end);
y=y_train(1:mround*kfold)';

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
perm = randperm(m);

bandwidth_all = [1:0.2:6,10:100];
count= 1;
for bandwidth = bandwidth_all
    CV_error = 0; %this will store the sum of squared error
    for i = 1:kfold
        testRange = mround*(i-1)+1:mround*i; %indicies for leave out set
        
        trainRange = 1:m;
        trainRange(testRange) = []; %training indicies
        
        Xtest_CV = X(perm(testRange),:);
        ytest_CV = y(perm(testRange));
        
        Xtrain_CV = X(perm(trainRange),:);
        ytrain_CV = y(perm(trainRange));
        
        [mTest,nTest]=size(Xtest_CV);
        [mTrain,nTrain]=size(Xtrain_CV);
                
        [Xtrain_CV,mu, s] = normalizeVars(Xtrain_CV);
        for j = 1:nTest %normalize test data
            Xtest_CV(:,j) =  Xtest_CV(:,j)-mu(j);
             Xtest_CV(:,j) =  Xtest_CV(:,j)./s(j);
        end
        
        for k = 1:mTest %loop through test examples and compute the sum or squared error
            % computing weights for example i
            temp = Xtrain_CV-repmat(Xtest_CV(k,:),mTrain,1);
            for j = 1:mTrain
                w(j) = temp(j,:)*temp(j,:)';
            end
            w = exp(-w/(2*bandwidth^2));
            
            mdl = fitlm(Xtrain_CV,ytrain_CV,'linear','weights',w);
            ypred = predict(mdl,Xtest_CV(k,:));         % use model to predict on new test data
            CV_error =CV_error +(ytest_CV(k)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error.
        end
    end
    CV_error_all(count) = CV_error/m;
    count = count+1;
end

fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(bandwidth_all,CV_error_all,'linewidth',1)
ylab = ylabel('CV');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('$\tau$');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
print('./Figures/eps/bandWidthTesting','-depsc')
print('./Figures/jpegs/bandWidthTesting','-djpeg','-r600')