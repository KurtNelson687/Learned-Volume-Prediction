function criterion = Lassofit(X_train,y_train,X_test,y_test)

%Normalize variables
[~,nTest]=size(X_test);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

[B,FitInfo] = lasso(X_train,y_train,'CV',10);      % train and create a linear regression model
ypred = X_test*B(:,FitInfo.Index1SE)+FitInfo.Intercept(FitInfo.Index1SE);           % use model to predict on new test data
criterion = sum((y_test-ypred).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end

