function [jval, gradientvec]= mla_your_neuon_network_cost_function(thetavec, X, y, lambda = 0)
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
[m, k_output_layer] = size(y);
[m, n_input_layer] = size(X);
n_hidden_layer = ceil(n_input_layer * 1.2);
num_hidden_layer = 1;		% set default as 1 to save computing power

% unroll theta into theta_input, theta_hidden and theta_output
num_elements = n_hidden_layer * (n_input_layer + 1) ;
theta_input = reshape(thetavec(1:num_elements), n_hidden_layer,n_input_layer + 1);
current_pos = num_elements;

theta_hidden = []
for i = 1: num_hidden_layer - 1
	num_elements = n_hidden_layer * (n_hidden_layer + 1) ;
	theta_hidden = [theta_hidden; reshape(thetavec(current_pos + 1:current_pos + num_elements), n_hidden_layer,n_hidden_layer + 1)];
	current_pos += num_elements;
end;

num_elements = k_output_layer * (n_hidden_layer + 1) ;
theta_output = reshape(thetavec(current_pos + 1:current_pos + num_elements), n_output_layer,n_hidden_layer + 1);

% start to compute forward propagation
% store result into a_input, a_hidden_matrix and a_output
% x0 is NOT in a_input, a_hidden_matrix or a_output

a_input = X;
a_input_with_bias = (ones(m,1),X);
z_input = a_input_with_bias * theta_input';
a_hidden = mla_logistic_regression_sigmoid(z_input);

a_hidden_matrix = [a_hidden];

for i = 1: num_hidden_layer - 1
	a_hidden = (ones(m,1),a_hidden);
	theta_this_layer = theta_hidden( (i-1)*n_hidden_layer + 1: i * n_hidden_layer, :);
	z_hidden = a_hidden * theta_this_layer';
	a_hidden = mla_logistic_regression_sigmoid(z_hidden);
	a_hidden_matrix = [a_hidden_matrix; a_hidden];
end;

a_hidden = (ones(m,1),a_hidden);
z_hidden = a_hidden * theta_output';
a_output = mla_logistic_regression_sigmoid(z_hidden);





% step 2: compute cost
% store cost into jval
cost_matrix = - y .* log(a_output) - ( 1-y ) .* log(1-a_output);
if lambda ~= 0
	theta_input_square = sum(sum(theta_input(:,2:end) .^ 2));
	theta_hidden_square = sum(sum(theta_hidden(:,2:end) .^ 2));
	theta_output_square = sum(sum(theta_output_square(:,2:end) .^ 2));
	theta_all_square_sum = theta_input_square_sum + theta_hidden_square_sum + theta_output_square_sum;
	regularization_factor = lambda / (2 * m) * theta_all_square_sum;
else
	regularization_factor = 0;
end;
jval = 1 / m * sum(sum(cost_matrix)) + regularization_factor;

%% step 3: backward propagation to get gradient

% compute diff
% store data into diff_output and diff_hidden_matrix

diff_output = a_output - y;
a_part = a_hidden_matrix(end - m + 1:end,:);
current_pos = end - m;
a_part = a_part .* ( 1 - a_part);
diff_last_layer = (diff_output * theta_output)(:,2:end) .* a_part;
diff_hidden_matrix = [diff_last_layer];

for i = 1: num_hidden_layer - 1
	a_part = a_hidden_matrix(current_pos - m + 1:current_pos,:);
	current_pos = current_pos - m;
	a_part = a_part .* ( 1 - a_part);
	theta_this_layer = theta_hidden(end - n_hidden_layer*i + 1:end - n_hidden_layer*(i-1),:);
	diff_last_layer = (diff_last_layer * theta_this_layer)(:,2:end) .* a_part;
	diff_hidden_matrix = [diff_last_laster; diff_hidden_matrix ];
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

for i = 1: num_hidden_layer - 1
	diff_next_layer = diff_hidden_matrix(current_pos +1:current_pos + m, :);
	a_this_layer = a_hidden_matrix( (i-1) * m + 1: (i-1) * m + m, :);
	penalty_this_layer = penalty_hidden ( (i-1) * n_hidden_layer + 1: (i-1) * n_hidden_layer + n_hidden_layer, :);
	deriv_theta_this_layer = 1 / m * diff_next_layer' * [ones(m,1), a_this_layer] + penalty_this_layer;
	current_pos += m;
end;

a_last_hidden_layer = a_hidden_matrix(end - m + 1: end, :);
deriv_theta_output = 1 / m * diff_output' * [ones(m,1), a_last_hidden_layer] + penalty_output;

% convert deriv matrix into vec
gradientvec = [theta_input(:), theta_hidden(:), theta_output(:)];

%% ==================== gradient checking =========================
%% Please turn off the checking flag once implemtatnion is checked

if gradient_checking_flag
	n = length( thetavec );
	epsilon = 0.00004;
	for i = 1:n
		theta_plus = thetavec;
		theta_plus(i) = theta_plus(i) + epsilon;
		theta_minus = thetavec;
		theta_minus(i) = theta_minus(i) - epsilon;
		grad_approx(i) = (mla_your_neuon_network_cost(theta_plus) - mla_your_neuon_network_cost(theta_minus)) / ( 2 * epsilon);

	end;
	%% check if gradientvec is around grad_approx; 
end;

end

