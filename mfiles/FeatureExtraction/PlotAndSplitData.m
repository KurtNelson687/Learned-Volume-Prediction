% Script: PlotAndSplitData.m
%
% Author: Kurt Nelson and Sam Maticka
%
% Purpose: This script removes bad data, creates plots to visulaize the
% data, then splits good data into training and test data and saves it.
%
% Outputs:
% 1) data.mat - contains test and training data
%%
close all; clear;
data_folder = '../../DataFiles/'; % path to where processed data file features are
load([data_folder 'FitData_All.mat']) % data containing features and actual volumes for each pour
load([data_folder 'movieLabel_All.mat']) % movie labels for QC purposes
cuppcm3 = .00422675; % conversion for cups per cm^3

saveGoodData = false; % choose whether to save reduced data set

% BadData gives movie numbers that will be removed. These indicies indicate
% outliers or features that were erroneously extracted. This should be
% automated in the future. 
badData = [8443; 8446; 8447; 8457;...
    8463; 8467; 8476; 8486; 8489; 8490;...
    8510; 8511; 8513; 8520; 8502;...
    8524; 8530; 8538; 8559; 8571; 8376;
    8528; 8529; 8733; 8734; 8735;...
    8658; 8662; 8666; 8669; 8671; 8672;...
    8674; 8686; 8695; 8699;8704; 8706;...
    8708; 8709; 8710; 8724; 8725; 8729; 
    8738; 8740; 8741; 8746; 8747; 8750; 8737];
%% Data Cleaning - finds indicies for good and bad data
count =1;
count2 = 1;
for i = 1:length(movieLabel)
    if isempty(find(badData==movieLabel(i),1))
        goodInd(count)=i;
        count=count+1;
    else
        badInd(count2)=i;
        count2 = count2+1;
    end
end


%% Create subsets of data, useable and nonusable, and fill bad values with NaN in "X" and "y"
% Rejected data
Xbad = X(badInd,:);
ybad = y(badInd);

% Kept data
Xgood = X(goodInd,:);
ygood = y(goodInd);
MovLabelgood = movieLabel(goodInd);

%Replace bad data with NaN's in original X and Y - keep NaN values in though to retrain
%proper indexing for identifying bad features.
X(badInd,:) = NaN;
y(badInd)   = NaN;

%% Compute physics based prediction
Vol     = pi/4*X(:,4)*cuppcm3; % extract the volume feature and convert it to cups
scale = nanmean(y'./Vol); % Scaling for physics prediction
yPhs  = scale*Vol; % physics based prediction (i.e. LSR with 3-way interaction only)

featureDescription = {'Duration (s)','Speed (cm/s)','L^2_{ave} (cm^2)',...
    'L^2*Speed*Duration'};

%% Feature plots against label
fig1 = figure;
for i = 1:size(X,2)-1
    subplot(5,1,i);plot(X(:,i),'k');ylabel(featureDescription{i});
end
subplot(5,1,4);plot((Vol'-y)./y,'k');ylabel('% Error for Vol')
subplot(5,1,5);plot((yPhs'-y)./y,'k');ylabel('% Error for Scaled Vol')
xlabel('Movie Number')

fig2 = figure;
for i = 1:size(X,2)
    subplot(2,2,i);plot(y,X(:,i),'*k');hold on;ylabel(featureDescription{i});
    plot(ybad,Xbad(:,i),'*r');
end
legend('Good Data','Bad Data','First Filming','location','NW')
xlabel('Actual Volume (cups)')

%% Simple data fitting to see how OLS is doing
%Removing NaN's for Ordinary least squares prediction
yPhs = yPhs(goodInd);
Xgood(:,end) = pi/4*Xgood(:,end)*cuppcm3; % convert units for 3-way interaction
Xgood = [ones(size(Xgood,1),1),Xgood]; % add column of ones to Xgood

%Ordinary least squares optimum theta
theta = inv(Xgood'*Xgood)*Xgood'*ygood';

yhat = Xgood*theta;

%Percent error calculations
OLSerror = abs(yhat - ygood')./ygood'*100;
meanOLSerror = mean(OLSerror);

physicsError = abs(yPhs - ygood')./ygood'*100;
meanPhysicsError = mean(physicsError);


%% Prediction plots
fig3 = figure;
plot(ygood,yhat,'.b',ygood,yPhs,'.r',[0, 2.5],[0 2.5],'k','markersize',10)
xlabel('Measured Volume (cups)'); ylabel('Predicted Volume (cups)')
leg = legend('OLS prediction','Physics');

%% plot Features against each other

figure;
cnt = 0;
for i = 1:4
    for j = 1:4
        cnt = cnt+1;
        subplot(4,4,cnt);plot(Xgood(:,i+1),Xgood(:,j+1),'.k'); % X1 is intercept.
        xlabel(['X',num2str(i)])
        ylabel(['X',num2str(j)])
    end
end


%% split and save test and training data

%sort data
[ygood, ind] = sort(ygood);
Xgood = Xgood(ind,:);
uniqueY = unique(ygood);

% Find two random indices for each unique volume tested
for i = 1:length(uniqueY)
     numSample(i) = length(find(y == uniqueY(i)));
     tempInd = randsample(numSample(i),2)+find(y == uniqueY(i),1, 'first')-1;
     if i == 1
        testInd = tempInd;
     else
         testInd = [testInd; tempInd];
     end
end

%Split data into traning and test data
trainInd = true(length(y),1);
trainInd(testInd)=false;
X_test = X(testInd,:);
y_test = y(testInd);
movieLabel_test = movieLabel(testInd);
X_train = X(trainInd,:);
y_train = y(trainInd);
movieLabel_train = movieLabel(trainInd);

if saveGoodData
    save([data_folder 'data.mat'],'X_train','y_train','X_test','y_test','movieLabel_train','movieLabel_test')
end



