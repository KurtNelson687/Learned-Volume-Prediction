function criterion = Lassofit(X_train,y_train,X_test,y_test)
[B,FitInfo] = lasso(X_train,y_train,'CV',10);      % train and create a linear regression model
ypred = X_test*B(:,FitInfo.Index1SE)+FitInfo.Intercept(FitInfo.Index1SE);           % use model to predict on new test data
criterion = sum((y_test-ypred).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end

