function [jVal, gradient]= mla_logistic_regression_cost_function(theta, X, y, lambda = 0)
%% Purpose: this function is to proivde cost function for fminunc 
%% Purpose: this function is about logistic regression
%% Info: lambda is used for regularization purpose

[m,n] = size(X);

% compute hyphothesis first
z = X * theta;
h = mla_logistic_regression_sigmoid(z);

% regularization
if lambda == 0 
	regu_factor_cost = 0;
else
	theta_square = theta .* theta;
	theta_square(1,1) = 0;
	regu_factor_cost = lambda / (2 * m) * sum(theta_square);
end;

% compute cost
jVal = mean(-y .* log(h) - ( 1-y ) .* log(1-h)) + regu_factor_cost;

% compute gradient
diff = h - y;

% regularization
if lambda == 0 
	regu_factor_deriv = zeros(n,1);
else
	regu_factor_deriv = lambda / m * theta;
	regu_factor_deriv(1,1) = 0;
end;

extended_diff = [];
for i = 1:n
	extended_diff = [extended_diff,diff];
end;
gradient = ( mean(X .* extended_diff) )' + regu_factor_deriv;

end

