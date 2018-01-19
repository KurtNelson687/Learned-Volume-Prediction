function criterion = KNNfit_reg3(X_train,y_train,X_test,y_test)

[mTest,nTest]=size(X_test);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest %normalize test data
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

% find the nearest neighbor in X_train for each point in X_test
Ind = knnsearch(X_train,X_test,'k',3);

ypred = y_train(Ind);

ypred = mean(ypred,2); % change mean dimension depending on k

criterion = sum((y_test(:)-ypred(:)).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end
