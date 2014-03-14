function test_logistic()
%% Purpose: this function is to test logistic sigmoid and cost_function algorithm

%% Initialization
clear ; close all; clc

% Load Data

data = load('test_data_ex2data1.mlt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================


[m, n] = size(X);

% Add x0
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = mla_logistic_regression_cost_function(initial_theta, X, y);

printf('Cost at initial theta (zeros): %f\n', cost);
printf('Gradient at initial theta (zeros): \n%f\n', grad);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(mla_logistic_regression_cost_function(t, X, y)), initial_theta, options);

% Print theta to screen
printf('Cost at theta found by fminunc: %f\n', cost);
printf('theta: \n');
printf(' %f \n', theta);

prob = mla_logistic_regression_sigmoid([1 45 85] * theta);
printf(['For a student with scores 45 and 85, we predict an admission probability of %f\n\n'], prob);

end