function gradientvec = mla_neuon_network_backward_propagation(X, y, lambda, ...
				a_input, a_hidden_matrix, a_output, 
				theta_input, theta_hidden, theta_output,
				n_input_layer, n_hidden_layer, n_output_layer, num_hidden_layer)
%% Purpose: this function perform backward propagation and return vec

m = size(y,1);

% comput diff first
diff_output = a_output - y;
a_part = a_hidden_matrix(end - m + 1:end,:);
current_pos = size(a_hidden_matrix,1) - m;
a_part = a_part .* ( 1 - a_part);
diff_last_layer = (diff_output * theta_output)(:,2:end) .* a_part;
diff_hidden_matrix = [diff_last_layer];

for i = 1: num_hidden_layer - 1
	a_part = a_hidden_matrix(current_pos - m + 1:current_pos,:);
	current_pos -= m;
	a_part = a_part .* ( 1 - a_part);
	theta_this_layer = theta_hidden(end - n_hidden_layer*i + 1:end - n_hidden_layer*(i-1),:);
	diff_last_layer = (diff_last_layer * theta_this_layer)(:,2:end) .* a_part;
	diff_hidden_matrix = [diff_last_layer; diff_hidden_matrix ];
end;


% compute deriv (forward as diff already computed)
% use below variables
% a_input, a_hidden_matrix, diff_hidden_matrix, diff_output
% theta_input, theta_hidden, theta_output
% lambda, m


if lambda ~= 0
	
	penalty_input = lambda/m * theta_input;
	penalty_output = lambda/m * theta_output;
	penalty_hidden = lambda/m * theta_hidden;
	penalty_input(:,1) = 0;	% no penalty for theta0
	penalty_output(:,1) = 0;	% no penalty for theta0
	penalty_hidden(:,1) = 0;	% no penalty for theta0
else
	penalty_input = 0;
	penalty_output = 0;
	penalty_hidden = 0;
end;

diff_first_hidden_layer = diff_hidden_matrix(1:m,:);
deriv_theta_input = 1 / m * diff_first_hidden_layer' * [ones(m,1), a_input] + penalty_input;
current_pos = m;

deriv_theta_hidden = [];
a_current_pos = 0;
for i = 1: num_hidden_layer - 1
	diff_next_layer = diff_hidden_matrix(current_pos +1:current_pos + m, :);
	current_pos += m;
	if lambda == 0 
		penalty_this_layer = 0;
	else
		penalty_this_layer = penalty_hidden( (i-1) * n_hidden_layer + 1: i * n_hidden_layer, :);
	end;
	a_this_layer = a_hidden_matrix(1 + a_current_pos: m + a_current_pos,:);
	a_current_pos += m;
	deriv_theta_this_layer = 1 / m * diff_next_layer' * [ones(m,1), a_this_layer] + penalty_this_layer;
	deriv_theta_hidden = [deriv_theta_hidden; deriv_theta_this_layer];
end;

a_last_hidden_layer = a_hidden_matrix(a_current_pos + 1: a_current_pos + m, :);
deriv_theta_output = 1 / m * diff_output' * [ones(m,1), a_last_hidden_layer] + penalty_output;


deriv_theta_hidden_vec = [];
deriv_pos = 0;
for i = 1: num_hidden_layer - 1
	deriv_theta_hidden_vec = [deriv_theta_hidden_vec; deriv_theta_hidden(deriv_pos + 1: deriv_pos + n_hidden_layer,:)(:)];
	deriv_pos += n_hidden_layer;
end;
% convert deriv matrix into vec
gradientvec = [deriv_theta_input(:); deriv_theta_hidden_vec; deriv_theta_output(:)];

end