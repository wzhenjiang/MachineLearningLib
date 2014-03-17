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

diff_output = a_output - y;
a_part = a_hidden_matrix(end - m + 1:end,:);
current_pos = size(a_hidden_matrix,1) - m;
a_part = a_part .* ( 1 - a_part);
diff_last_layer = (diff_output * theta_output)(:,2:end) .* a_part;
diff_hidden_matrix = [diff_last_layer];

for i = 1: num_hidden_layer - 1
	a_part = a_hidden_matrix(current_pos - m + 1:current_pos,:);
	current_pos = current_pos - m;
	a_part = a_part .* ( 1 - a_part);
	theta_this_layer = theta_hidden(end - n_hidden_layer*i + 1:end - n_hidden_layer*(i-1),:);
	diff_last_layer = (diff_last_layer * theta_this_layer)(:,2:end) .* a_part;
	diff_hidden_matrix = [diff_last_layer; diff_hidden_matrix ];
end;


% compute deriv (forward as diff already computed)
% use below variables
% a_input, a_hidden_matrix, diff_hidden_matrix, diff_output
% theta_input, theta_hidden, theta_output
% lambda, m


if lambda ~= 0
	
	penalty_input = lambda * theta_input;
	penalty_output = lambda * theta_output;
	penalty_hidden = lambda * theta_hidden;
	penalty_input(:,1) = 0;	% no penalty for theta0
	penalty_output(:,1) = 0;	% no penalty for theta0
	penalty_hidden(:,1) = 0;	% no penalty for theta0
else
	penalty_input = 0;
	penalty_output = 0;
	penalty_hidden = 0;
end;

diff_first_hidden_layer = diff_hidden_matrix(1:m,:);
deriv_theta_input = 1 / m * diff_first_hidden_layer' * [ones(m,1), a_input] + penalty_input;
current_pos = m;

deriv_theta_hidden = [];
a_current_pos = 0;
for i = 1: num_hidden_layer - 1
	diff_next_layer = diff_hidden_matrix(current_pos +1:current_pos + m, :);
	current_pos += m;
	if lambda == 0 
		penalty_this_layer = 0;
	else
		penalty_this_layer = penalty_hidden( (i-1) * n_hidden_layer + 1: (i-1) * n_hidden_layer + n_hidden_layer, :);
	end;
	a_this_layer = a_hidden_matrix(1 + a_current_pos: m + a_current_pos,:);
	a_current_pos += m;
	deriv_theta_this_layer = 1 / m * diff_next_layer' * [ones(m,1), a_this_layer] + penalty_this_layer;
	deriv_theta_hidden = [deriv_theta_hidden; deriv_theta_this_layer];
end;

a_last_hidden_layer = a_hidden_matrix(a_current_pos + 1: a_current_pos + m, :);
deriv_theta_output = 1 / m * diff_output' * [ones(m,1), a_last_hidden_layer] + penalty_output;

% convert deriv matrix into vec
gradientvec = [deriv_theta_input(:); deriv_theta_hidden(:); deriv_theta_output(:)];

%% ==================== gradient checking =========================
%% Please turn off the checking flag once implemtatnion is checked

if gradient_checking_flag
	grad_thetavec = mla_neuon_network_generate_numercial_gradient(X, y, thetavec, ...
							n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer, lambda);
	diff_grad = grad_thetavec - gradientvec;
	index = find(diff_grad > 0.001);
	if size(index,1) > 0
		printf('numerical checking failed\n');
		exit();
	end;
end;

end

