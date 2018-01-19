close all; clear all;
load('../DataFiles/data.mat')
addpath('./functions');
numTest = 1000;
X = X_train(:,4);
y = y_train;
uniqueY = unique(y);

for trial = 1:numTest
       
    % Resplit data finding two random indices for each unique volume tested
    for i = 1:length(uniqueY)
        numSample(i) = length(find(y == uniqueY(i)));
        tempInd = randsample(numSample(i),2)+find(y == uniqueY(i),1, 'first')-1;
        if i == 1
            testInd = tempInd;
        else
            testInd = [testInd; tempInd];
        end
    end
    
    % Split data into traning and test data
    trainInd = true(length(y),1);
    trainInd(testInd)=false;
    X_test = X(testInd,:);
    y_test = y(testInd);
    
    X_train = X(trainInd,:);
    y_train = y(trainInd);
    
    error_all(trial) = PhysMdl_log(X_train,y_train,X_test,y_test)/length(testInd);
end
MeanError = mean(error_all);
RMSError  = sqrt(MeanError);