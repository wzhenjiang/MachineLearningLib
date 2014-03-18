function test_neuon_network()
%% Purpose: this function is used to test neuon network functions

clear ; close all; clc

%% initialize required parameters
n_input_layer = 400;		% 20 * 20 images
n_hidden_layer = 25;		% 25 hidden units
n_output_layer = 10;		% 1 - 10 as output result
num_hidden_layer = 1;		

%% =========== Part 1: Loading and Visualizing Data =============

% Load Sample Data
printf('Loading and Visualizing Data ...\n')

% X and y are loaded
load('test_data_neuon_network_sample.mat');
m = size(X, 1);

y_matrix = zeros(m,10);
for i=1:m
	y_matrix(i,y(i)) = 1;
end;
y = y_matrix;

% Randomly select 100 data points to display
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% test_plot_neuon_network_sample_data(X(sel, :));

fprintf('Sample data loaded and converted!\n');
fflush(stdout);

%% ================ Part 2: Loading Parameters ================

printf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('test_data_neuon_network_weight.mat');

theta_input = Theta1;
theta_output = Theta2;
theta_hidden = [];

% Unroll parameters 
theta_vec = [theta_input(:) ; theta_output(:)];

printf('\nNeural Network Parameters loaded and unrolled!\n')


%% ================ Part 3: Compute Cost (Feedforward) ================

printf('Size of X: %d * %d \n', size(X,1),size(X,2));
printf('Size of y: %d * %d \n', size(y,1),size(y,2));
printf('Size of nn_params: %d * %d \n', size(theta_vec,1),size(theta_vec,2));
fflush(stdout);

printf('\nForward propagation of Neural Network ...\n')

lambda = 0;

printf('Computing cost ...');
fflush(stdout);

[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
											theta_input, theta_hidden, theta_output, ...
											1, 25);

cost = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);

printf(['Cost at parameters (loaded from weights): %f '...
         '\n(this value should be about 0.287629)\n'], cost);


% Weight regularization parameter (we set this to 1 here).
printf('\nChecking Cost Function (w/ Regularization) ... \n')
lambda = 1;

cost = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);

printf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], cost);

lambda = 3;
cost = mla_neuon_network_compute_cost(a_output,y, lambda, theta_input, theta_hidden, theta_output);

printf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 3): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], cost);

printf('\nEvaluating sigmoid gradient...\n')

g = mla_logistic_regression_sigmoid([1 -0.5 0 0.5 1]);
printf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
printf('%f ', g);
printf('\n\n');


%% ==================== start to check cost function ==================================
printf('\nChecking Backward Propagation... \n');

test_neuon_network_check_gradient(0);

printf('\nBackward propagation checked.\n');

printf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
test_neuon_network_check_gradient(lambda);
printf('\nBackward propagation checked.\n');
fflush(stdout);



%%  ======================== start to train ============================
printf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset( 'MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

printf('\nInitializing Neural Network Parameters ...\n')

% theta_vec = mla_neuon_network_theta_initialization(n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer);

printf('\nStart to train ...\n')

cost_func = @(p) mla_neuon_network_cost_function(p, X, y, lambda, n_hidden_layer, num_hidden_layer);
% [theta_result, cost] = fmincg(cost_func, theta_vec, options);

%options = optimset( 'MaxIter', 5, 'GradObj', 'on');
%[theta_result, cost] = fminunc(cost_func, theta_vec, options);

max_iteration = 400;
theta_result = theta_vec;
printf('\n');
for i = 1:max_iteration
	[jval, gradientvec] = mla_neuon_network_cost_function(theta_result, X, y, lambda, n_hidden_layer, num_hidden_layer);
	theta_result = theta_result - gradientvec;
	printf('\rIteration [%i] cost is [%f] ...',i, jval);
	fflush(stdout);
end;
printf('\n');

printf('theta data:\n');
disp([theta_vec(1:10),theta_result(1:10)]);

% Obtain Theta1 and Theta2 back from nn_params
theta_input = reshape(theta_result(1:n_hidden_layer * (n_input_layer + 1)), ...
                 n_hidden_layer, (n_input_layer + 1));

theta_output = reshape(theta_result((1 + (n_hidden_layer * (n_input_layer + 1))):end), ...
                 n_output_layer, (n_hidden_layer + 1));

printf('Training is done.\n');

[a_input, a_hidden_matrix, a_output] = mla_neuon_network_forward_propagation(X, ...
											theta_input, [], theta_output, ...
											num_hidden_layer, n_hidden_layer);
[m,n] = size(a_output);
for i = 1:m
	for j = 1:n
		if a_output(i,j) >= 0.5
			a_output(i,j) = 1;
		else
			a_output(i,j) = 0;
		end;
	end;
end;

[accu,n] = size(find(sum(a_output - y,2) ~= 0)) ;

printf('\nTraining Set Accuracy: %f\n', 100 - accu / m * 100);

end

