function test_logistic()
%% Purpose: this function is to test logistic sigmoid and cost_function algorithm

%% Initialization
clear ; close all; clc

% Load Data

data = load('test_data_ex2data1.mlt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Test logistic regression ====================


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

printf('start fminunc...\n');
fflush(stdout);
%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(mla_logistic_regression_cost_function(t, X, y)), initial_theta, options);

% Print theta to screen
printf('Cost at theta found by fminunc: %f\n', cost);
printf('theta: \n');
printf(' %f \n', theta);

prob = mla_logistic_regression_sigmoid([1 45 85] * theta);
printf(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob);

z = X * theta;
h = mla_logistic_regression_sigmoid(z);

for i = 1:length(y)
	if h(i) >= 0.5
		h(i) = 1;
	else
		h(i) = 0;
	end;
end
printf('Train Accuracy: %f\n', mean(double(h == y)) * 100);


%% ==================== Part 2: Test logistic regression with regularization ====================

data = load('test_data_ex2data2.mlt');
X = data(:, [1, 2]); y = data(:, 3);

X1 = X(:,1);
X2 = X(:,2);
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end
X = out;

initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = mla_logistic_regression_cost_function(initial_theta, X, y, lambda);

printf('Cost at initial theta (zeros): %f\n', cost);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

printf('start fminunc...\n');
fflush(stdout);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(mla_logistic_regression_cost_function(t, X, y, lambda)), initial_theta, options);


% Compute accuracy on our training set
z = X * theta;
h = mla_logistic_regression_sigmoid(z);

for i = 1:length(y)
	if h(i) >= 0.5
		h(i) = 1;
	else
		h(i) = 0;
	end;
end

fprintf('Train Accuracy with lambda: %f\n', mean(double(h == y)) * 100);



end