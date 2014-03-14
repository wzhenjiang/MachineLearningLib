function [jVal, gradient]= mla_logistic_regression_cost_function(theta, X, y)
%% Purpose: this function is to proivde cost function for fminunc 
%% Purpose: this function is about logistic regression

% compute hyphothesis first
z = X * theta;
h = mla_logistic_regression_sigmoid(z);

% compute cost
jVal = mean(-y .* log(h) - ( 1-y ) .* log(1-h));

% compute gradient
diff = h - y;
[m,n] = size(X);
extended_diff = [];
for i = 1:n
	extended_diff = [extended_diff,diff];
end;
gradient = ( mean(X .* extended_diff) )';

end

