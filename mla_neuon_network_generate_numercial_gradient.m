function grad_thetavec = mla_neuon_network_generate_numercial_gradient(X, y, thetavec, ...
							n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer, lambda)
%% Purpose: this function is to generate numerical gradient for checking purpose

n = length( thetavec );
epsilon = 1e-4;

grad_thetavec = [];

for i = 1:n
	% compute cost of theta_plus
	theta_plus = thetavec;
	theta_plus(i) = theta_plus(i) + epsilon;
	[theta_input, theta_hidden, theta_output] = mla_neuon_network_roll_thetavec(theta_plus, ...
								n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer);
	[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
												theta_input, theta_hidden, theta_output, ...
												num_hidden_layer, n_hidden_layer);
	cost_plus = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);
	
	% compute cost of theta_minus
	theta_minus = thetavec;
	theta_minus(i) = theta_minus(i) - epsilon;
	[theta_input, theta_hidden, theta_output] = mla_neuon_network_roll_thetavec(theta_minus, ...
								n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer);
	[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
												theta_input, theta_hidden, theta_output, ...
												num_hidden_layer, n_hidden_layer);
	cost_minus = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);
	
	% comput grad
	grad_thetavec = [grad_thetavec; (cost_plus - cost_minus) / ( 2 * epsilon)];

end;

end