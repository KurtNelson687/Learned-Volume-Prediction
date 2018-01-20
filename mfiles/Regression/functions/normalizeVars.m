function [varsOut,mu, s] = normalizeVars(varsIn)
% Function: normalizeVars
%
% Author: Kurt Nelson and Sam Maticka
%
% Purpose: %This function removes the column mean from the columns of varsIn,
% scales the standard deviation so its 1, and returns the column mean,
% standard deviation, and normalized variables,
% 
% Input:
% 1) varsIn - variables to normalize
%
% Output:
% 1) varsOut - normalized variables
% 2) mu - mean vector
% 3) s - standard deviations
%%

[~,n]=size(varsIn);
mu = mean(varsIn); % comput column means
for i = 1:n
    varsOut(:,i) = varsIn(:,i)-mu(i); % subtract out mean
end
s = sqrt(mean(varsOut.^2)); % compute standard deviations now with mean zero
for i = 1:n
    varsOut(:,i) = varsOut(:,i)./s(i); % normalize
end
end

