function [theta_output, cost_history] = mla_linear_regression_gradient_descent(X, y, theta, alpha = 0.1, iteration = 10, history = false)
%% Purpose:		Generate new_theta for one step further
%% Purpose:		This function is part of gradient descent algorithm
%% Attention:	X0 should be added in X already

theta_output = theta;

cost_history = [];

if history
	cost_history = [mla_linear_regression_cost(X,y,theta)];
end;

for iter = 1: iteration
	% compute hypothesis first
	h = X * theta_output;

	[m,n] = size(X);

	% compute diff of the two vectors
	diff = h - y;

	% extend diff in order to .* X
	extended_diff = [];
	for i = 1: n
		extended_diff = [extended_diff ,diff];
	end;

	% compute delta
	delta = ( alpha / m * sum ( X .* extended_diff ) )';

	% update theta simultaneously
	theta_output = theta_output - delta;
	if history
		cost_history = [cost_history; mla_linear_regression_cost(X,y,theta_output)];
	end;

end;

end
