function criterion = KNNfit6(X_train,y_train,X_test,y_test)
[X_train,y_train,X_test,y_test] = combineThenSplit(X_train,y_train,X_test,y_test);

%normalize data
[mTest,nTest]=size(X_test);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest 
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

mdl = fitcknn(X_train,y_train,'Distance','euclidean','BreakTies','nearest');
mdl.NumNeighbors = 6;

[ypred,~,~] = predict(mdl,X_test);

criterion = sum(y_test(:)~=ypred(:));
end

