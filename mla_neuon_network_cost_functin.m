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
gradient_checking_flag = true;
%% ===========================================================

% First, please use reshape to get back your theta matrix
% theta = reshape(thetavec(a,b), m,n);

[m,n] = size(X);

% step 1: forward propagation to get hypothesis
% in this section, we assume all hidden layers has same number of units

[m, k_output_layer] = size(y);
[m, n_input_layer] = size(X);
n_hidden_layer = ceil(n_input_layer * 1.2);
num_hidden_layer = 1;		% set default as 1 to save computing power

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
theta_input = reshape(thetavec(current_pos + 1:current_pos + num_elements), n_output_layer,n_hidden_layer + 1);

a_input = (ones(m,1),X);
z_input = a_input * theta_input';
a_hidden = mla_logistic_regression_sigmoid(z_input);

a_hidden_matrix = [a_hidden];

for i = 1: num_hidden_layer - 1
	a_hidden = (ones(m,1),a_hidden);
	z_hidden = a_hidden * theta_hidden';
	a_hidden = mla_logistic_regression_sigmoid(z_hidden);
	a_hidden_matrix = [a_hidden_matrix; a_hidden];
end;

a_hidden = (ones(m,1),a_hidden);
z_hidden = a_hidden * theta_output';
a_output = mla_logistic_regression_sigmoid(z_hidden);





% step 2: compute cost
cost_matrix = - y .* log(a_output) - ( 1-y ) .* log(1-a_output);
theta_factor = lambda / (2 * m) * sum(sum(a_hidden_matrix(:,2:end) .^ 2))+ sum(sum(theta_output(:,2:end) .^ 2));
cost = 1 / m * sum(sum(cost_matrix)) + theta_factor;

% step 3: backward propagation to get gradient


% compute hyphothesis first
z = X * theta;
h = mla_logistic_regression_sigmoid(z);

% regularization
if lambda == 0 
	regu_factor_cost = 0;
else
	theta_square = theta .* theta;
	theta_square(1,1) = 0;
	regu_factor_cost = lambda / (2 * m) * sum(theta_square);
end;

% compute cost
jVal = mean(-y .* log(h) - ( 1-y ) .* log(1-h)) + regu_factor_cost;

% compute gradient
diff = h - y;

% regularization
if lambda == 0 
	regu_factor_deriv = zeros(n,1);
else
	regu_factor_deriv = lambda / m * theta;
	regu_factor_deriv(1,1) = 0;
end;

extended_diff = [];
for i = 1:n
	extended_diff = [extended_diff,diff];
end;
gradient = ( mean(X .* extended_diff) )' + regu_factor_deriv;

% finally, convert gradient into gradientvec

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

