function [theta_input, theta_hidden, theta_output] = mla_neuon_network_roll_thetavec(thetavec, n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer)
%% Purpose: this function is used to unroll thetavec into theta

num_elements = n_hidden_layer * (n_input_layer + 1) ;
theta_input = reshape(thetavec(1:num_elements), n_hidden_layer,n_input_layer + 1);
current_pos = num_elements;

theta_hidden = [];
for i = 1: num_hidden_layer - 1
	num_elements = n_hidden_layer * (n_hidden_layer + 1) ;
	theta_hidden = [theta_hidden; reshape(thetavec(current_pos + 1:current_pos + num_elements), n_hidden_layer,n_hidden_layer + 1)];
	current_pos += num_elements;
end;

num_elements = n_output_layer * (n_hidden_layer + 1) ;
theta_output = reshape(thetavec(current_pos + 1:current_pos + num_elements), n_output_layer,n_hidden_layer + 1);

end
