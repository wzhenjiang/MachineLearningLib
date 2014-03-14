function cost = mla_linear_regression_cost(X, y, theta, lambda = 0)
%% Purpose:	This function is to compute cost
%% Info:	Square error is used here as it applies to most of cases
%% Info:	J(theta) = 1 / (2*m) * sum( (h(x)-y) .^ 2 )
%% Info: lambda is used for regularization purpose

% compute h
h = X * theta;
[m,n] = size(X);

% regularization
if lambda == 0 
	regu_factor_cost = 0;
else
	theta_square = theta .* theta;
	theta_square(1,1) = 0;
	regu_factor_cost = lambda / (2 * m) * sum(theta_square);
end;

% compute cost
cost = 1 / (2*m) * sum( (h-y) .^ 2 ) + regu_factor_cost;

end
