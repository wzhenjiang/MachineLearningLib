function [a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, theta_input, theta_hidden, theta_output, num_hidden_layer, n_hidden_layer)
%% Purpose: forward propagation of neuon network to comput activation of all neuon units

[m,n] = size(X);
a_input = X;
a_input_with_bias = [ones(m,1),X];
z_input = a_input_with_bias * theta_input';
a_hidden = mla_logistic_regression_sigmoid(z_input);

a_hidden_matrix = [a_hidden];

for i = 1: num_hidden_layer - 1
	a_hidden = [ones(m,1),a_hidden];
	theta_this_layer = theta_hidden( (i-1)*n_hidden_layer + 1: i * n_hidden_layer, :);
	z_hidden = a_hidden * theta_this_layer';
	a_hidden = mla_logistic_regression_sigmoid(z_hidden);
	a_hidden_matrix = [a_hidden_matrix; a_hidden];
end;

a_hidden = [ones(m,1),a_hidden];
z_hidden = a_hidden * theta_output';
a_output = mla_logistic_regression_sigmoid(z_hidden);

end