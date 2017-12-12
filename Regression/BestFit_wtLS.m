%% This script uses 10-fold cross validation to test the bandwidth parameter for local
% weighted regression.

close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');

%% Perform Feature selection using linear regression and LOOCV
% Model feature selection performing k-fold cross validation
numKfold = 10;      % LOOCV: k=m
featuresIn = [1,2,3]; % features forced to be included in fit (Hierarchical principle)
opts = statset('display','iter'); % displays iterations 

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

[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

[fs1,history1] = sequentialfs(@wt_local_fit,X_train,y_train,'cv',numKfold,...
        'keepin',featuresIn,'nfeatures',nAll,'options',opts);    
[minCV, indMinCV] = min(history1.Crit);

X(:,history1.In(indMinCV,:));

 mdl = fitlm(X_train(:,history1.In(indMinCV,:)),y_train,'linear','weights',w);
 ypred = predict(mdl,X_test(:,history1.In(indMinCV,:)));         % use model to predict on new test data
 criterion =criterion +(y_test(i)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error. 

X_test = X(:,history1.In(indMinCV,:)); %Extracting only features we want

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