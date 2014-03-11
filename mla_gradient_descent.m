function new_theta = mla_gradient_descent(theta, X, y, alpha = 0.1)
%% Purpose: generate new_theta for one step further
%% Purpose: this funciton is part of gradient descent algrithm
%% Attention: X0 should be added in X already

h = X * theta;

[m,n] = size(X);

new_theta = zero(n,1);

% compute diff
diff = h - y;

% extend diff to .* X
for i = 1: n-1
    diff = [diff ,diff];
end;

% compute delta
delta = ( alpha / m * sum ( X .* diff ) )';

% update theta simultaneously
new_theta = theta - delta;


end