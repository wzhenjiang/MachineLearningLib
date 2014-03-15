
% h should be a m * k matrix, y should be m * k matrix as well
partial_cost = mean (sum (- y .* log(h) + (1 - y) .* log(1-h),2));

% theta_factor = theta .^ 2 of all layers except theta0
j = partial_cost + lambda / (2 * m) * theta_factor
