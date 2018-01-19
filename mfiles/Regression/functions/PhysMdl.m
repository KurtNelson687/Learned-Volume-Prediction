function criterion = PhysMdl(X_train,y_train,X_test,y_test)
    scale = nanmean(y_train)./nanmean(X_train); %Scaling for physics prediction
    ypred = scale*X_test;
    
    criterion = sum((y_test-ypred).^2);    % specify the criterion to evaluate model performance. sum of squared error.
end

