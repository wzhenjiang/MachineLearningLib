function cost = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output)
%% Purpose: this function is used to compute cost of neuon network

% store cost into jval

[m, k] = size(y);

cost_matrix = - y .* log(a_output) - ( 1-y ) .* log(1-a_output);
if lambda ~= 0
	theta_input_square_sum = sum(sum(theta_input(:,2:end) .^ 2));
	theta_hidden_square_sum = sum(sum(theta_hidden(:,2:end) .^ 2));
	theta_output_square_sum = sum(sum(theta_output(:,2:end) .^ 2));
	theta_all_square_sum = theta_input_square_sum + theta_hidden_square_sum + theta_output_square_sum;
	regularization_factor = lambda / (2 * m) * theta_all_square_sum;
else
	regularization_factor = 0;
end;

cost = 1 / m * sum(sum(cost_matrix))  + regularization_factor;

end