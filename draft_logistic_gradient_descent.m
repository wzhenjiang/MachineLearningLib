function [jVal, gradient]= mla_logistic_cost_function(theta, X, y, alpha = 0.01, iteration = 10)

%% Purpose: this function is to calculate optimal theta
%% Purpose: this function is about logistic

optimal_theta = theta;

for iter = 1:iteration
	diff = mla_logistic_sigmoid(optimal_theta, X) - y;
	delta = sum([diff, diff] .* X)';
	% update theta simultaneously
	optimal_theta = optimal_theta - alpha * delta;
end;

end



