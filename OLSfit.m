function criterion = OLSfit(X_train,y_train,X_test,y_test)
mdl = fitlm(X_train,y_train,'linear'); % train and create a linear regression model
ypred = predict(mdl,X_test);           % use model to predict on new test data
criterion = sum((y_test-ypred).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end

