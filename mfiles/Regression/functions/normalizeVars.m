function [varsOut,mu, s] = normalizeVars(varsIn)
%This function removes the column mean from the columns of varsIn,
%scales the standard deviation so its 1, and returns the column mean,
%standard deviation, and normalized variables 
[~,n]=size(varsIn);
mu = mean(varsIn);
for i = 1:n
    varsOut(:,i) = varsIn(:,i)-mu(i);
end
s = sqrt(mean(varsOut.^2));
for i = 1:n
    varsOut(:,i) = varsOut(:,i)./s(i);
end
end

