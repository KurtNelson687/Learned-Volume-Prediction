function criterion = wt_local_fit(X_train,y_train,X_test,y_test)
% Function: wt_percentDiff
%
% Purpose: Fit weighted least squares on the train data. Then test on test
% data. The function outputs the sum of squared errors.
%
% Inputs:
% 1) X_train - training data features
% 2) y_train - training data response
% 3) X_test - test data features
% 4) y_test - test data response

% Outputs:
% 1) criterion - sum of squared error on the test data
%%
bandwidth = 2.2; %bandwidth for weighted least squares. This was determined from bandwidth testing
%% setup data
[mTest,nTest]=size(X_test);
[mTrain,nTrain]=size(X_train);
%Normalize variables
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

%% Fite model
criterion = 0; %this will store the sum of squared error
for i = 1:mTest %loop through test examples and compute the sum or squared error
    
    % computing weights for example i
    temp = X_train-repmat(X_test(i,:),mTrain,1);
    for j = 1:mTrain
        w(j) = temp(j,:)*temp(j,:)';
    end
    w = exp(-w/(2*bandwidth^2));
    
    mdl = fitlm(X_train,y_train,'linear','weights',w);
    ypred = predict(mdl,X_test(i,:));         % use model to predict on new test data
    criterion =criterion +(y_test(i)-ypred).^2;  % Compute sum of squared error. 
end

end

