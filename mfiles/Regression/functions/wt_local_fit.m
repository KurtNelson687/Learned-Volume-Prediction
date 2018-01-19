% weighted least squares fit
function criterion = wt_local_fit(X_train,y_train,X_test,y_test)
% train and create a linear regression model using the expected value as the weight
bandwidth = 2.2;
[mTest,nTest]=size(X_test);
[mTrain,nTrain]=size(X_train);

criterion = 0; %this will store the sum of squared error

[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

for i = 1:mTest %loop through test examples and compute the sum or squared error
    
    % computing weights for example i
    temp = X_train-repmat(X_test(i,:),mTrain,1);
    for j = 1:mTrain
        w(j) = temp(j,:)*temp(j,:)';
    end
    w = exp(-w/(2*bandwidth^2));
    
    mdl = fitlm(X_train,y_train,'linear','weights',w);
    ypred = predict(mdl,X_test(i,:));         % use model to predict on new test data
    criterion =criterion +(y_test(i)-ypred).^2;  % specify the criterion to evaluate model performance. sum of squared error. 
end

end

