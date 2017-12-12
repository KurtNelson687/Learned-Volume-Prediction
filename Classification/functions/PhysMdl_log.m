function criterion = PhysMdl_log(X_train,y_train,X_test,y_test)
    scale = nanmean(y_train./X_train'); %Scaling for physics prediction
    ypred = scale*X_test;
    
    ypred = round(ypred*4)/4;
    criterion = sum(y_test(:)~=ypred(:));    % specify the criterion to evaluate model performance. sum of squared error.
end

