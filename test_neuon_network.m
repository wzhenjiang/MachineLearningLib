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

lambda = 3;
cost = mla_neuon_network_compute_cost(a_output,y, lambda, Theta1, [], Theta2);

printf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 3): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], cost);

fprintf('\nEvaluating sigmoid gradient...\n')

g = mla_logistic_regression_sigmoid([1 -0.5 0 0.5 1]);
printf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
printf('%f ', g);
printf('\n\n');

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_nn_params = mla_neuon_network_theta_initialization(input_layer_size, hidden_layer_size, num_labels, 1);


%% ==================== start to check cost function ==================================
printf('\nChecking Backward Propagation... \n');

test_neuon_network_check_gradient(0);

printf('\nBackward propagation checked.\n');

printf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
[X,y,theta_input, theta_hidden, theta_output,num_hidden_layer, n_hidden_layer] = test_neuon_network_check_gradient(lambda);
printf('\nBackward propagation checked.\n');

%%  ======================== start to train ============================
printf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;



end

