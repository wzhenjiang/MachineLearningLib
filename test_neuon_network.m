function test_neuon_network()
%% Purpose: this function is used to test neuon network functions

clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
printf('Loading and Visualizing Data ...\n')

load('test_data_neuon_network_sample.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

% test_plot_neuon_network_sample_data(X(sel, :));

fprintf('Sample data rendered!\n');
fflush(stdout);

%% ================ Part 2: Loading Parameters ================

printf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('test_data_neuon_network_weight.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================

printf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;
y_matrix = zeros(m,10);
for i=1:m
	y_matrix(i,y(i)) = 1;
end;
y = y_matrix;
printf('Size of X: %d * %d \n', size(X,1),size(X,2));
printf('Size of y: %d * %d \n', size(y,1),size(y,2));
printf('Size of nn_params: %d * %d \n', size(nn_params,1),size(nn_params,2));
fflush(stdout);

printf('start to compute cost ...');
fflush(stdout);

[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
											Theta1, [], Theta2, ...
											1, 25);

cost = mla_neuon_network_compute_cost(a_output,y, lambda, Theta1, [], Theta2);

printf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], cost);

printf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

cost = mla_neuon_network_compute_cost(a_output,y, lambda, Theta1, [], Theta2);

printf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], cost);


fprintf('\nEvaluating sigmoid gradient...\n')

g = mla_logistic_regression_sigmoid([1 -0.5 0 0.5 1]);
printf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
printf('%f ', g);
printf('\n\n');

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_nn_params = mla_neuon_network_theta_initialization(input_layer_size, hidden_layer_size, num_labels, 1);


%% ==================== start to check cost function ==================================
printf('\nChecking Backward Propagation... \n');

lambda = 0;

n_input_layer = 3;
n_hidden_layer = 5;
n_output_layer = 3;
num_hidden_layer = 3;
m = 5;

W = zeros(n_hidden_layer, 1 + n_input_layer);
Theta1 = reshape(sin(1:numel(W)), size(W)) / 10;

W = zeros(n_hidden_layer, 1 + n_hidden_layer);
Thetax = reshape(sin(1:numel(W)), size(W)) / 10;
Theta_X = [Thetax;Thetax];

W = zeros(n_output_layer, 1 + n_hidden_layer);
Theta2 = reshape(sin(1:numel(W)), size(W)) / 10;
W = zeros(m, n_input_layer);  %-1?
X = reshape(sin(1:numel(W)), size(W)) / 10;

y  = 1 + mod(1:m, n_output_layer)';
y_matrix = zeros(m,n_output_layer);
for i=1:m
	y_matrix(i,y(i)) = 1;
end;
y = y_matrix;

% Unroll parameters
nn_params = [Theta1(:) ; Theta_X(:); Theta2(:)];

% Short hand for cost function
cost_func = @(p) mla_neuon_network_cost_function(p, X, y, lambda, n_hidden_layer, num_hidden_layer);
printf('call cost function\n');
[cost, grad] = cost_func(nn_params);
printf('done\n');
numgrad = mla_neuon_network_generate_numercial_gradient(X, y, nn_params, ...
							n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer, lambda);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
k = [1: 98]';
disp([k numgrad grad]);

diff = numgrad - grad;
index = find(diff > 0.001);
printf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

printf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);


printf('\nBackward propagation checked.\n');

index






end

