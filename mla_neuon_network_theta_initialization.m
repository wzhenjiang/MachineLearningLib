function thetavec = mla_neuon_network_theta_initialization(n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer)
%% Purpose: initialize thetavec with random number

%% Info: initial_theta generated is in [-init_range, init_range]
init_range = sqrt(6) / sqrt(n_input_layer + n_output_layer);

theta_input = zeros(n_hidden_layer, n_input_layer + 1);
theta_input = rand(n_hidden_layer,n_input_layer + 1) * ( 2 * init_range ) - init_range;

theta_hidden_matrix = [];
for i = 1: num_hidden_layer - 1
	theta_hidden = zeros(n_hidden_layer, n_hidden_layer + 1);
	theta_hidden = rand(n_hidden_layer,n_hidden_layer + 1) * ( 2 * init_range ) - init_range;
	theta_hidden_matrix = [theta_hidden_matrix; theta_hidden];
end;

theta_output = zeros(n_output_layer, n_hidden_layer + 1);
theta_output = rand(n_output_layer,n_hidden_layer + 1) * ( 2 * init_range ) - init_range;

thetavec = [theta_input(:); theta_hidden_matrix(:); theta_output(:)];



end