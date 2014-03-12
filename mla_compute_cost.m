function cost = mla_compute_cost(X, y, theta)
%% Purpose:	This function is to compute cost
%% Info:	Square error is used here as it applies to most of cases
%% Info:	J(theta) = 1 / (2*m) * sum( (h(x)-y) .^ 2 )

% compute h
h = X * theta;
[m,n] = size(X);

% compute cost
cost = 1 / (2*m) * sum( (h-y) .^ 2 );

end
