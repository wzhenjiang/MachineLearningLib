function [X,y,theta_input, theta_hidden, theta_output,num_hidden_layer, n_hidden_layer] = test_neuon_network_check_gradient(lambda = 0)
%% Purpose: this function is used to verify if algorithm is implemented correctly

n_input_layer = 3;
n_hidden_layer = 5;
n_output_layer = 3;
num_hidden_layer = 1;
m = 5;

W = zeros(n_hidden_layer, 1 + n_input_layer);
theta_input = reshape(sin(1:numel(W)), size(W)) / 10;

W = zeros(n_hidden_layer, 1 + n_hidden_layer);
Thetax = reshape(sin(1:numel(W)), size(W)) / 10;
theta_hidden = [];

W = zeros(n_output_layer, 1 + n_hidden_layer);
theta_output = reshape(sin(1:numel(W)), size(W)) / 10;
W = zeros(m, n_input_layer);  %-1?
X = reshape(sin(1:numel(W)), size(W)) / 10;

y  = 1 + mod(1:m, n_output_layer)';
y_matrix = zeros(m,n_output_layer);
for i=1:m
	y_matrix(i,y(i)) = 1;
end;
y = y_matrix;

% Unroll parameters
nn_params = [theta_input(:) ; theta_hidden; theta_output(:)];

% Short hand for cost function
cost_func = @(p) mla_neuon_network_cost_function(p, X, y, lambda, n_hidden_layer, num_hidden_layer);

[cost, grad] = cost_func(nn_params);

numgrad = mla_neuon_network_generate_numercial_gradient(X, y, nn_params, ...
							n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer, lambda);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
k = [1: 38]';

% disp([numgrad grad]);
printf('data display disabled!\n');

printf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

printf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end