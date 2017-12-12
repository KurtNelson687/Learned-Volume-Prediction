% this script compares the error for using a different number of neighbors
% to compare against. 3 neighbors seems to be the best. I ran it multiple
% times.
clear;clc;close all;
load('../DataFiles/data.mat')
addpath('./functions');
numK = 1:8;
numTrial = 1:200;
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

for trial=numTrial
    trial
    for i=numK
        if i==1
            model = @KNNfit_reg;
        elseif i==2
            model = @KNNfit_reg2;
        elseif i==3
            model = @KNNfit_reg3;
        elseif i==4
            model = @KNNfit_reg4;
        elseif i==5
            model = @KNNfit_reg5;
        elseif i==6
            model = @KNNfit_reg6;
        elseif i==7
            model = @KNNfit_reg7;
        elseif i==8
            model = @KNNfit_reg8;
        end
        
        
        %% create possible feature sets
        % Rename and remove intercept feature
        
        % Perform feature selection and requires that all of the original features are included
        opts = statset('display','off');
        
        % Perform feature selection employing Hierarchical principle (keep features 1:3 in)
        [fs1,history1] = sequentialfs(model,X,y,'cv',numKfold,...
            'keepin',featuresIn,'nfeatures',n,'options',opts);
        sse1 = min(history1.Crit);  % sum of squared error from the best fit (sse=mse, since 1 test point)
        
        
        % Compute the mean squared error predicted. the lowest of the 4 is the k chosen
        minCV(trial,i) = min(history1.Crit);
    end
end

meanMinCV = mean(minCV);

fig1 = figure;%('visible', 'off');
fig1.PaperUnits = 'centimeters';
fig1.PaperPosition = [0 0 8 4];
set(gca,'box','on')
plot(numK(2:end),meanMinCV(2:end),'linewidth',1)
ylab = ylabel('CV');
set(ylab,'interpreter','Latex','FontSize',8)
xlab = xlabel('Number of Neighbors');
set(xlab,'interpreter','Latex','FontSize',8)
set(gca,'FontSize',6)
print('./Figures/eps/neighborTesting','-depsc')
print('./Figures/jpegs/neighborTesting','-djpeg','-r600')