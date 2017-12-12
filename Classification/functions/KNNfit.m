function criterion = KNNfit(X_train,y_train,X_test,y_test)
%This combines data then resamples so that 2 examples from each label are
%considered. Sequentialfs is nolonger performing true cross validation.
%Instead it is using bootstrapping without replacement. 

[X_train,y_train,X_test,y_test] = combineThenSplit(X_train,y_train,X_test,y_test);

%normalize data
[mTest,nTest]=size(X_test);
[X_train,mu, s] = normalizeVars(X_train);
for i = 1:nTest 
    X_test(:,i) = X_test(:,i)-mu(i);
    X_test(:,i) = X_test(:,i)./s(i);
end

mdl = fitcknn(X_train,y_train,'Distance','euclidean','BreakTies','nearest');
mdl.NumNeighbors = 1;
[ypred,~,~] = predict(mdl,X_test);

criterion = sum(y_test(:)~=ypred(:));
end

