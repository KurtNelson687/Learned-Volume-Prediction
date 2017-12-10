function criterion = KNNfit(X_train,y_train,X_test,y_test)
mdl = fitcknn(X_train,y_train,'Distance','euclidean');
mdl.NumNeighbors = 4;

[ypred,~,~] = predict(mdl,X_test);

criterion = sum(y_test(:)~=ypred(:));
end

