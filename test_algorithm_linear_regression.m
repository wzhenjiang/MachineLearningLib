function test_algorithm_linear_regression()
%% Purpose: this function is to test if mla_(gradient_descent, normal_equation, feature_normalization, compute_cost) works right
%% Purpose: this function is to test if mlp_(scatter) works right


clear ; close all; clc

%% =================== Part 1: test mla_gradient_descent ===================
data = load('test_data_ex1data1.mlt');
iteration = 1500;
alpha = 0.01;

X = data(:,1);
X = [ones(size(X,1),1),X];
y = data(:,2);
theta = zeros(2,1);
cost = mla_compute_cost(X, y, theta);
printf('Cost of initial theta: %f\n', cost);

theta = mla_gradient_descent(X, y, theta, alpha, iteration);
cost = mla_compute_cost(X, y, theta);
printf('Cost of final theta: %f\n', cost);

% print theta to screen
printf('Theta found by mla_gradient_descent: [%f %f ]\n', 
			theta(1), theta(2));

% Predict values for population sizes of 35,000 and 70,000
prediction = [1, 3.5] * theta;
printf('Profit prediction result of population = 35,000 : [%f]\n',
    prediction*10000);
	
prediction = [1, 7] * theta;
printf('Profit prediction result of population = 70,000 : [%f]\n',
    prediction*10000);
fflush(stdout);



%% =================== Part 2: test mla_normal_euquation ===================
data = load('test_data_ex1data1.mlt');

X = data(:,1);
X = [ones(size(X,1),1),X];
y = data(:,2);

theta = mla_normal_equation(X, y);
cost = mla_compute_cost(X, y, theta);
printf('Cost of final theta: %f\n', cost);

% print theta to screen
printf('Theta found by mla_normal_equation: [%f %f ]\n', 
			theta(1), theta(2));

% Predict values for population sizes of 35,000 and 70,000
prediction = [1, 3.5] * theta;
printf('Profit prediction result of population = 35,000 : [%f]\n',
    prediction*10000);
	
prediction = [1, 7] * theta;
printf('Profit prediction result of population = 70,000 : [%f]\n',
    prediction*10000);
fflush(stdout);

	
%% =================== Part 3: test mla_feature_normalization ===================

data = load('test_data_ex1data2.mlt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
printf('First 10 examples from the dataset: \n');
printf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fflush(stdout);

[X mu sigma] = mla_mean_normalization(X);

% Print out Normalization result
printf('First 10 examples from normalized result: \n');
printf(' x = [%.5f %.5f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fflush(stdout);

% Add X0
X = [ones(m, 1) X];

alpha = 0.01;
num_iters = 400;

theta = zeros(3, 1);
[theta, J_history] = mla_gradient_descent(X, y, theta, alpha, num_iters, true);
printf('Theta computed from the normal equations: \n');
printf(' %f \n', theta);

[X mu sigma] = mla_mean_normalization([1650, 3], mu, sigma);
X = [1, X];

price = X * theta; 
printf('Price projected with [1650,3]: %f \n', price);


data = load('test_data_ex1data2.mlt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add X0
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = mla_normal_equation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);

X = [1 1650 3];

price = X * theta; 
printf('Price projected with [1650,3]: %f \n', price);


	
end
