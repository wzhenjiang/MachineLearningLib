function [jval, gradientvec]= mla_neuon_network_cost_function(thetavec, X, y, lambda = 0, n_hidden_layer, num_hidden_layer = 1)
%% Purpose: this function is to proivde cost function for fminunc 
%% Purpose: this function is about neuon network
%% Info: this function must be customized per your neuon network
%% Attention: gradientvec and thetavec here are vector
%% Info: to create thetavec, please use thetavec = [theta1(:); theta2(:); ...]
%% Info: inital thetavec should be randomly generated to get nw work
%% Info: initial_theta = rand(m,n)*(2*init_range) - init_range;
%% Info: initial_theta generated is in [-init_range, init_range]

%% ===== this section is the control of gradient checking ====
%% please set gradient_checking_flag to false once imp checked
gradient_checking_flag = false;
%% ===========================================================

% First, please use reshape to get back your theta matrix
% theta = reshape(thetavec(a,b), m,n);

[m,n] = size(X);

%% step 1: forward propagation to get hypothesis
% in this section, we assume all hidden layers has same number of units

% check out structure of the neuon network
[m, n_output_layer] = size(y);
[m, n_input_layer] = size(X);
if ~exist('n_hidden_layer','var')
	n_hidden_layer = ceil(n_input_layer * 1.2);
end;

% unroll theta into theta_input, theta_hidden and theta_output
[theta_input, theta_hidden, theta_output] = mla_neuon_network_roll_thetavec(thetavec, ...
								n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer);

% start to compute forward propagation
% store result into a_input, a_hidden_matrix and a_output
% x0 is NOT in a_input, a_hidden_matrix or a_output

[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
											theta_input, theta_hidden, theta_output, ...
											num_hidden_layer, n_hidden_layer);


% step 2: compute cost
% store cost into jval
jval = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);

%% step 3: backward propagation to get gradient

% compute diff
% store data into diff_output and diff_hidden_matrix

gradientvec = mla_neuon_network_backward_propagation(X, y, lambda, ...
				a_input, a_hidden_matrix, a_output, 
				theta_input, theta_hidden, theta_output,
				n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer);

%% ==================== gradient checking =========================
%% Please turn off the checking flag once implemtatnion is checked


if gradient_checking_flag
	grad_thetavec = mla_neuon_network_generate_numercial_gradient(X, y, thetavec, ...
							n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer, lambda);
	diff_grad = norm(grad_thetavec - gradientvec) / norm(grad_thetavec + gradientvec);
	if diff_grad > 1e-6
		printf('numerical checking failed\n');
		exit();
	end;
end;

end

